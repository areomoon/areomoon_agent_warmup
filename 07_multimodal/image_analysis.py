"""
Multimodal Analysis of Scientific Figures
==========================================

Scientific papers contain four modalities requiring different extraction strategies:
  1. Text     — methods, results sections (direct LLM reading)
  2. Tables   — experimental data, often with merged cells (vision + parsing)
  3. Images   — SEM/TEM micrographs, crystal structures (vision LLM)
  4. Plots    — XRD patterns, I-V curves, spectroscopy (vision + chart parsing)

This module demonstrates vision LLM extraction for figures in materials papers.
Key challenge: describe quantitative information (peak positions, axis values)
from plots that can be validated against text claims.

References:
  - ColPali (arXiv 2407.01449): https://arxiv.org/abs/2407.01449
  - DocLLM (arXiv 2401.00908): https://arxiv.org/abs/2401.00908
  - Claude API Vision: https://docs.anthropic.com/en/docs/build-with-claude/vision
"""

import os
import json
import base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def encode_image(image_path: str) -> str:
    """Encode image to base64 for API calls."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_sem_image(image_path: str, client) -> dict:
    """
    Analyze a SEM/TEM micrograph to extract morphological parameters.
    Returns: particle size, shape, distribution, scale bar reading.
    """
    img_b64 = encode_image(image_path)
    prompt = """Analyze this SEM/TEM micrograph from a materials science paper.
Extract:
1. Particle/feature morphology (shape: spherical, rod, plate, irregular)
2. Estimated size range (use the scale bar)
3. Size distribution (monodisperse/polydisperse)
4. Surface features (porosity, faceting, agglomeration)
5. Scale bar value and unit (if visible)

Return JSON:
{
  "morphology": "...",
  "size_range_nm": [min, max],
  "size_distribution": "monodisperse|narrow|broad",
  "surface_features": "...",
  "scale_bar_nm": null or float,
  "confidence": 0.0-1.0,
  "notes": "any uncertainty or special observations"
}"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ]
        }],
        response_format={"type": "json_object"},
        max_tokens=500,
    )
    return json.loads(response.choices[0].message.content)


def analyze_xrd_pattern(image_path: str, material: str, client) -> dict:
    """
    Extract peak positions and phase identification from XRD pattern image.
    """
    img_b64 = encode_image(image_path)
    prompt = f"""Analyze this XRD (X-ray diffraction) pattern for {material}.

Extract:
1. Major peak positions (2θ degrees)
2. Phase identification (crystal structure, e.g., cubic perovskite, wurtzite)
3. Presence of secondary phases
4. Estimated peak FWHM (for crystallite size estimation)
5. Any anomalies (peak splitting, broadening)

Return JSON:
{{
  "material": "{material}",
  "crystal_structure": "...",
  "major_peaks_2theta": [list of floats],
  "secondary_phases": [],
  "phase_pure": true/false,
  "notes": "...",
  "confidence": 0.0-1.0
}}"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ]
        }],
        response_format={"type": "json_object"},
        max_tokens=500,
    )
    return json.loads(response.choices[0].message.content)


def analyze_property_plot(image_path: str, plot_description: str, client) -> dict:
    """
    Extract quantitative data from property measurement plots
    (resistivity vs T, I-V curves, spectroscopy, etc.)
    """
    img_b64 = encode_image(image_path)
    prompt = f"""Analyze this {plot_description} from a materials science paper.

Extract:
1. X-axis: label, range, units
2. Y-axis: label, range, units
3. Key data points (transitions, peaks, plateaus)
4. Trend description
5. Any labeled values or annotations in the figure

Return JSON:
{{
  "x_axis": {{"label": "...", "range": [min, max], "unit": "..."}},
  "y_axis": {{"label": "...", "range": [min, max], "unit": "..."}},
  "key_features": [
    {{"type": "transition|peak|plateau", "x_value": ..., "y_value": ..., "description": "..."}}
  ],
  "trend": "...",
  "confidence": 0.0-1.0
}}"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ]
        }],
        response_format={"type": "json_object"},
        max_tokens=600,
    )
    return json.loads(response.choices[0].message.content)


def analyze_full_paper_page(page_image_path: str, client) -> dict:
    """
    Analyze a full paper page image (PDF page rendered as image).
    Identifies all figures, tables, and text blocks.
    Maps to the ColPali approach: use vision model for whole-page retrieval.
    """
    img_b64 = encode_image(page_image_path)
    prompt = """Analyze this scientific paper page. Identify and describe:

1. Text blocks (sections, paragraphs)
2. Figures (type: SEM, XRD, plot, schematic; location on page)
3. Tables (column headers, data type)
4. Chemical formulas or equations

Extract any numerical values visible in figures or tables.

Return JSON:
{
  "page_content_types": ["text", "figure", "table"],
  "figures": [{"type": "...", "caption": "...", "key_values_visible": [...]}],
  "tables": [{"headers": [...], "data_type": "..."}],
  "key_numerical_values": [{"value": ..., "unit": "...", "context": "..."}]
}"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ]
        }],
        response_format={"type": "json_object"},
        max_tokens=800,
    )
    return json.loads(response.choices[0].message.content)


def demo_without_image():
    """Demonstrate the API structure without requiring actual images."""
    print("=== Multimodal Extraction Demo ===\n")
    print("Place scientific figure images in data/figures/ to run real analysis.\n")

    print("Available analysis functions:")
    functions = [
        ("analyze_sem_image()", "SEM/TEM → particle size, morphology, scale bar"),
        ("analyze_xrd_pattern()", "XRD pattern → peak positions, phase identification"),
        ("analyze_property_plot()", "Property plots → key transitions, axis values"),
        ("analyze_full_paper_page()", "Full PDF page → all figures, tables, values"),
    ]
    for fn, desc in functions:
        print(f"  {fn}")
        print(f"    → {desc}\n")

    print("Materials science figure types and what to extract:")
    figure_types = {
        "SEM/TEM": ["particle size (nm)", "morphology", "scale bar", "agglomeration"],
        "XRD": ["2θ peak positions", "crystal structure", "phase purity", "FWHM"],
        "Resistivity vs T": ["T_MI or T_c (K)", "metallic/insulating behavior", "ΔR/R"],
        "I-V or P-V": ["Voc (V)", "Isc (mA/cm²)", "PCE (%)", "FF"],
        "Raman/FTIR": ["peak wavenumbers (cm⁻¹)", "phase identification", "D/G ratio"],
    }
    for fig_type, extractions in figure_types.items():
        print(f"  {fig_type}: {', '.join(extractions)}")


if __name__ == "__main__":
    demo_without_image()

    # Uncomment to run with actual images:
    # from openai import OpenAI
    # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # result = analyze_sem_image("data/figures/sample_sem.jpg", client)
    # print(json.dumps(result, indent=2))

# TODO: Add PDF → page images pipeline (pypdf + pillow)
# TODO: Add Gemini 1.5 Pro for 1M context PDF analysis (whole paper at once)
# TODO: Add table extraction with structure preservation (merged cells)
# TODO: Add cross-modal validation: XRD peak claims vs. image analysis

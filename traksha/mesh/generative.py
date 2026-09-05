"""What exists in single-image-to-3D city generation, and why none of it runs here.

This module holds no model and drives no subprocess. It is a **record**, kept in
code rather than in a paragraph someone has to trust, of the systems that have
been evaluated for this pipeline and what became of each.

It exists because the question keeps arriving in the same shape - "why is
Sat2City not wired in?" - and the answer is a fact about that project rather
than a gap in this one. `traksha doctor` reads this table, so the answer is one
command away and cannot drift from what the code does.

**None of these is a refiner.** Not one takes an input mesh: each reads an image
and predicts a representation, and geometry falls out of it. So none can sharpen
the geometry TRAKSHA measures - each can only propose a *different* geometry,
from the same single view, with no constraint tying it to the metric vertical
datum this pipeline spends its whole length establishing. Sat2City v2 says so
about itself: it scales every asset "isotropically to fit inside
[-0.5, 0.5]^3", which is a normalised cube and not metres.

**What is actually used instead is `mesh.trellis`.** Reading Sat2City v2's
architecture closely shows that exactly one of its modules is trained - the
geometry flow, the part that invents a shape - and every other stage is a frozen
TRELLIS.2 component. The trained one is the part this pipeline does not want.
The frozen ones are downloadable. So the appearance half of Sat2City v2 runs
here with no training at all, on our own measured mesh, and that is the refiner
this project ships. See README section 6.2g.

Status, checked on 2026-09-05 by reading the repositories rather than the
project pages:

    sat2city     unreleased  github.com/thua919/Sat2City-release holds exactly
                             one file, README.md, reading "Coming soon"
    sat2city-v2  unreleased  project page says "Code Coming"; no repository
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Provider:
    """One single-image-to-3D system, and what is actually known about it."""

    name: str
    title: str
    venue: str
    released: bool
    repo: Optional[str] = None
    checkpoint: Optional[str] = None
    licence: Optional[str] = None
    input_px: Optional[int] = None
    output: str = ""
    metric: bool = False
    unavailable_reason: str = ""
    notes: str = ""
    trained_on: str = ""

    def describe(self) -> str:
        if not self.released:
            return f"{self.title} ({self.venue}) - {self.unavailable_reason}"
        bits = [f"{self.title} ({self.venue})"]
        if self.checkpoint:
            bits.append(f"checkpoint {self.checkpoint}")
        if self.licence:
            bits.append(self.licence)
        bits.append("metric" if self.metric else "normalised, not metric")
        return ", ".join(bits)


PROVIDERS = {
    "sat2city": Provider(
        name="sat2city",
        title="Sat2City: 3D city generation from a single satellite image with "
              "cascaded latent diffusion",
        venue="ICCV 2025",
        released=False,
        repo="https://github.com/thua919/Sat2City-release",
        input_px=300,
        output="sparse voxel grid carrying geometry and appearance",
        metric=False,
        unavailable_reason=(
            "no code has been published. The release repository "
            "github.com/thua919/Sat2City-release contains exactly one file, "
            "README.md, whose entire content is 'Coming soon'. There is nothing "
            "to install and nothing to call."),
        notes="Conditions on a satellite-derived height map rather than on the "
              "image, so the input image cannot directly control texture - a "
              "limitation its own v2 paper raises against it.",
    ),
    "sat2city-v2": Provider(
        name="sat2city-v2",
        title="Sat2City v2: native 3D city asset generation from a single "
              "satellite image",
        venue="arXiv 2606.24138, June 2026",
        released=False,
        repo=None,
        input_px=512,
        output="textured mesh with PBR materials (base colour, metallic, "
               "roughness, alpha); 512^3 O-Voxel geometry, 2048x2048 textures",
        metric=False,
        trained_on="16,241 satellite-mesh pairs over 24 regions in 9 US cities, "
                   "from Google Earth 3D Tiles",
        unavailable_reason=(
            "no code and no checkpoint. The project page at "
            "ai4city-hkust.github.io/Sat2City-v2 says 'Code Coming' and names "
            "no repository; the v1 repository is a placeholder. Nothing to "
            "install and nothing to call."),
        notes="Explicitly non-metric: it scales every asset isotropically into "
              "[-0.5, 0.5]^3, and its own authors state that metric accuracy "
              "'would require more tightly coupled geospatial data'. Its "
              "appearance half needs none of the missing code: seven of its "
              "eight stages are frozen TRELLIS.2 modules and only its geometry "
              "flow is trained. `mesh.trellis` runs those seven on our own "
              "measured mesh, with no training. See README section 6.2g.",
    ),
}


def get(name: str) -> Provider:
    """One provider by name, or a ValueError that lists the alternatives."""
    key = str(name or "").strip().lower()
    if key in PROVIDERS:
        return PROVIDERS[key]
    raise ValueError(
        f"unknown generative provider '{name}'. Recorded: "
        + ", ".join(f"{k} ({'released' if v.released else 'unreleased'})"
                    for k, v in PROVIDERS.items()))


def released() -> list:
    """Providers that could actually be run today. Currently none."""
    return [p for p in PROVIDERS.values() if p.released]


def table() -> str:
    """The record as a block a reader can act on."""
    width = max(len(k) for k in PROVIDERS)
    rows = []
    for key, p in PROVIDERS.items():
        state = "released  " if p.released else "unreleased"
        rows.append(f"  {key:<{width}}  {state}  {p.describe()}")
    return "\n".join(rows)


def summary() -> str:
    """One line for `traksha doctor`."""
    runnable = [p.name for p in released()]
    if runnable:
        return f"{', '.join(runnable)} runnable of {len(PROVIDERS)} recorded"
    return (f"none of {len(PROVIDERS)} recorded systems has published code; "
            "the refiner in use is mesh.trellis")

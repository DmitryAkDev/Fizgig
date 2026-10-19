"""The Start tab's News card: newest first. Each item is a date, a title, a body (plain text, paragraphs split on a
blank line) and optional links ({"text", "href"}). The card shows the newest few; the full articles are docs/NEWS.md."""

_ARTICLE = "https://github.com/shootthesound/Fizgig/blob/master/docs/NEWS.md"

NEWS = [
    {
        "date": "20 Oct 2026",
        "title": "Kromoss out as a preview model",
        "body": (
            "Kromoss is no longer offered as the MOSS-Image preview model: it's too different from MOSS-Image Base for "
            "reliable previews of a LoRA trained on Base, and as a base to train on it overcooks. If you set Kromoss on "
            "the preview model or Base DiT row, press Download models for me in Preferences: it points both back to "
            "SFT and MOSS-Image Base.\n\n"
            "The 8-step Turbo LoRA stays, for fast previews of your Base: untick Use the SFT model for samples and set "
            "Turbo LoRA strength to 1 (12 steps)."
        ),
        "links": [{"text": "Read more", "href": _ARTICLE + "#20-oct-2026-kromoss-out-as-a-preview-model-the-turbo-lora-stays"}],
    },
    {
        "date": "19 Oct 2026",
        "title": "MOSS-Image training, fixed",
        "body": (
            "MOSS-Image LoRAs now learn likeness much sooner, and the picture holds its composition from epoch to "
            "epoch. MOSS's own schedule put only about 11% of training steps where faces and detail are learned; "
            "Fizgig's new Clean-end share puts 30% there. By eye the pictures are far sharper and more accurate, with "
            "no body horror. It's the default: there's nothing to set."
        ),
        "links": [{"text": "Read more", "href": _ARTICLE + "#19-oct-2026-moss-image-training-fixed"}],
    },
]

"""The Start tab's News card: newest first. Each item is a date, a title, a body (plain text, paragraphs split on a
blank line) and optional links ({"text", "href"}). The card shows the newest few; the full articles are docs/NEWS.md."""

_ARTICLE = "https://github.com/shootthesound/Fizgig/blob/master/docs/NEWS.md"

NEWS = [
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
    {
        "date": "19 Oct 2026",
        "title": "Kromoss and its 8-step Turbo LoRA",
        "body": (
            "Kromoss, lodestones' sharper fine-tune of MOSS-Image Base made for generating, can now be your preview "
            "model in place of SFT: pick it beside Download models for me in Preferences. You still train on Base: in "
            "our test a LoRA trained on Kromoss began to overbake after about 750 steps. Your previews then show the "
            "LoRA on Kromoss.\n\n"
            "Its Turbo LoRA (Download on the Turbo LoRA row) works on Kromoss previews: set Turbo LoRA strength to 1 on "
            "the Samples tab for 12-step previews. It's very good compared with the earlier MOSS-Image speed LoRAs; we "
            "suggest 12 steps rather than its card's 8."
        ),
        "links": [{"text": "Read more", "href": _ARTICLE + "#19-oct-2026-kromoss-and-its-8-step-turbo-lora"}],
    },
]

"""Prompt fragments used by the evolution agents."""


EDIT_CONTEXT: dict[str, str] = {
    "O1": (
        "This edit type can involve: City; Coast; Snow; Forest; Desert; Garden; Field; etc."
    ),
    "O2": (
        "This edit type can involve: Metal; Marble; Glass; Rubber; Fabric; Wood/bamboo; Ceramic; "
        "Wicker; etc."
    ),
    "O3": (
        "This edit type can involve: Animal replacements; Human replacements; Building "
        "replacements; Object and equipment replacements; etc."
    ),
    "O5": (
        "This edit type can involve: Animals; Facilities/tools; Props/decorations; "
        "Containers/tableware; Furniture; Lights; Food; Vehicles/buildings; Plants; "
        "Clothing/accessories; People; etc."
    ),
    "O6": (
        "This edit type can involve: Fibre crafts; Building blocks; Designer toys; Origami; "
        "Cartoon illustration; Watercolour; Oil painting; Pixel art; 3D animation; Comics; Paper "
        "cut; etc."
    ),
    "O7": (
        "This edit type can involve: Purple; Gold; Silver; Grey; Brown; White; Pink; "
        "Green/turquoise; Blue; Yellow; Black; Red; Orange; etc."
    ),
    "O8": (
        "This edit type can involve: Surface sheen; Surface roughness; Surface texture patterns; "
        "etc."
    ),
    "T1": (
        "This edit type can involve: Time and date information; Numeric information; Entity "
        "names; Fields and terminology; etc."
    ),
    "T3": (
        "This edit type can involve: Ranking labels; Data annotations; Process and status labels; "
        "Knowledge annotations; etc."
    ),
    "T4": (
        "This edit type can involve: English to Chinese; Chinese to English; English to Spanish; "
        "English to Japanese; etc."
    ),
    "T5": (
        "This edit type can involve: Title-block rewriting; Footer-block rewriting; Blurb and "
        "tagline rewriting; Brand-name rewriting; etc."
    ),
    "T6": (
        "This edit type can involve: Title repositioning; Footer and note repositioning; Relative "
        "text hierarchy; Badge repositioning; etc."
    ),
    "T7": (
        "This edit type can involve: Weight/slant; Line decorations; Typeface; Colour; etc."
    ),
    "H1": (
        "This edit type can involve: Body postures; Gestures; Task-related actions; Everyday "
        "object interactions; etc."
    ),
    "H2": (
        "This edit type can involve: Hairstyles; Clothing; Beard removal; Age changes; Facial "
        "expressions; etc."
    ),
    "H3": (
        "This edit type can involve: Skin smoothing; Wrinkle removal; Beard removal; etc."
    ),
    "G1": (
        "This edit type can involve: Comics; Oil painting; Ink painting; Ghibli; Pixar; Ukiyo-e; "
        "Watercolour; Pixel art; etc."
    ),
    "G2": (
        "This edit type can involve: Single-object repositioning; Position exchanges; Arrangement "
        "patterns; Grouped repositioning; etc."
    ),
    "G3": (
        "This edit type can involve: Forest environments; Urban environments; Functional indoor "
        "environments; Science-fiction environments; Neon night-city environments; Snowy "
        "mountain-cabin environments; etc."
    ),
    "G4": (
        "This edit type can involve: Brightness; Colour temperature; Time of day; Weather "
        "lighting; etc."
    ),
}

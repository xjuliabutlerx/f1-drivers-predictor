import streamlit as st

st.set_page_config(page_title="Model Naming Scheme", page_icon="🏁", layout="wide")

st.title("🏁 Model Naming Scheme")

st.write("Every generation of models in this project is named after a famous F1 driver, chosen to reflect that driver's on-track style in relation to how the model actually behaves - not just an arbitrary label. This naming convention mirrors `f1-constructors-predictor`'s track-based naming, just adapted to this project's own domain: drivers instead of circuits.")

st.info("**A note on v1 vs v2 naming:** v1 and v2 still mean what they normally mean - v1 is the foundational generation, v2 is the second. But every v2 keeper model beat every v1 model on rho, so the names were swapped between the two generations: the widely-cited GOAT names (Prost, Schumacher, Senna) now belong to v2's strongest results, and v1 took on v2's original names (Fangio, Clark, Button) in exchange. The version numbers track *which generation*; the names track *how good the specific model actually is*.")

st.header("v1")

st.subheader("Fangio 🇦🇷")
st.markdown("""
- The most precise and accurate of the v1 generation
- Sharp final-season accuracy, with the smallest maximum error of the three models
- Held out on 2020-2021, rho 0.9908
- Juan Manuel Fangio holds the highest win percentage of any driver in Formula 1 history - a fitting name for the sharpest model of this generation, even if it's since been surpassed by v2's own Prost model on the very same holdout seasons (rho 0.9938)
""")

st.subheader("Clark 🇬🇧")
st.markdown("""
- Steady, consistent accuracy gains all season long, with no dramatic swings up or down
- Strong overall, but defined by its evenness rather than any single standout jump in form
- Held out on 2022-2023, rho 0.9881
- Jim Clark's reputation was for an almost effortless, natural smoothness - dominant without ever looking like he was trying hard, which fits this model's steady, unremarkable-but-solid consistency
""")

st.subheader("Button 🇬🇧")
st.markdown("""
- The most volatile of the v1 generation - noticeably rougher early in a season, with the largest swings in error
- Still fundamentally strong by the end of a season, but carries a wider margin of error throughout than the other two
- Held out on 2024-2025, rho 0.9529
- Jenson Button's driving was defined by smooth, technical, calculated racecraft - a slightly ironic fit here, since this is actually the roughest of the three v1 models, but it's still the only model this project has for the 2024-2025 pair at all
""")

st.header("v2")

st.write("The v2 generation isn't a direct model-for-model upgrade of v1 - it's a small curated set of the strongest individual runs from a round of architecture experiments (wider hidden layers, plus testing ReLU against GELU and LeakyReLU activations), not one model per held-out season pair. Two of the three happen to share the same held-out seasons (2022-2023), evaluated with different activation functions, and there's no v2 model at all for 2024-2025 - none of the attempts on that pair beat v1's own result on it.")

st.subheader("Prost 🇫🇷")
st.markdown("""
- The standout of the entire v2 generation - the highest rho (0.9968), lowest error, and tightest max error (off by at most 1 position) of any model produced so far
- Held out on 2022-2023, using LeakyReLU activation
- Just as Alain Prost, "The Professor", built his reputation on calculated, low-risk, mistake-minimizing racing, this model is the most surgical of anything this project has produced - especially fitting for the single best result so far
""")

st.subheader("Schumacher 🇩🇪")
st.markdown("""
- The strongest all-around result among the "standard" v2 runs - tight, clean, low error - but not the most extreme standout of the generation
- Held out on 2020-2021, using the original ReLU activation, rho 0.9938 - beating v1's Fangio model on the very same seasons
- Mirrors Michael Schumacher's relentless, methodical consistency - a strong, dependable result without being the headline of the generation
""")

st.subheader("Senna 🇧🇷")
st.markdown("""
- The steadiest, most solid of the v2 generation rather than a standout - still excellent by any absolute measure (rho 0.9917), just the least remarkable relative to the other two
- Held out on 2022-2023, using GELU activation - the only difference from Prost is the activation function, which is what makes the gap between them notable
- Like Ayrton Senna, this name usually evokes genius-level highs - here it's attached to the most comparatively modest of the three v2 results, a reminder that these names track overall project standing, not a perfect behavioral match every time
""")

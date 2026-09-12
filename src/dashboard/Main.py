import streamlit as st

st.set_page_config(page_title="F1 Drivers Predictor", page_icon="🏎️", layout="wide")

st.title("🏎️ Formula 1 Drivers' Championship Ranking Model")
st.subheader("A Fan Project")

st.header("About This Project")
st.write("This project trains machine learning ranking models to predict Formula 1 Drivers' Championship standings from in-season data. It's the standalone sibling to my `f1-constructors-predictor` project, which does the equivalent for the Constructors' Championship. There's no shared code between the two, but the same overall approach where the two domains agree.")

st.markdown("""
What these models can do:
- Predict how the full driver field will finish, using only data available up through the current round
- Work regardless of how many drivers are on the grid in a given season (the model doesn't hardcode a field size)
- Account for mid-season driver swaps and substitutes, since not every driver has raced every round
""")

st.header("The Model Generations: In a Nutshell")
st.markdown("""
- **v1**: the foundational generation. A pairwise ranking approach validated by holding out whole seasons rather than individual rows, on the original 57-feature set
- **v2 (final generation)**: same features as v1, wider hidden layers (a validated win, tested in isolation), plus an activation-function experiment (ReLU vs. GELU vs. LeakyReLU). Every v2 model beat every v1 model on rho
- **v3**: attempted, not pursued. Momentum features, a driver-age feature, and a SHAP-driven feature prune were each tested in isolation against the same v2 architecture, and none produced a consistent improvement across all 3 held-out season pairs. See [Model Development & Training](/Model_Development_&_Training) for the full results
""")

st.header("Tech Stack")
st.markdown("""
- Python 3.11
- [`fastf1`](https://github.com/theOehrly/Fast-F1)
- torch (PyTorch)
- pandas
- numpy
- scipy
- scikit-learn
- matplotlib / seaborn
- shap
""")

st.html("For the full training and model architecture code, please see my GitHub repository <a href='https://github.com/xjuliabutlerx/f1-drivers-predictor' target='_blank'>f1-drivers-predictor</a>.")

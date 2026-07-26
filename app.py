# =========================================================
# RECOMMENDATIONS
# =========================================================

st.divider()

st.subheader("💡 Recommendations")

if "Healthy" in best_name:

    st.success(
        """
        🌿 The plant appears healthy based on the AI prediction.

        Recommended actions:
        • Continue regular watering and proper plant nutrition.
        • Monitor the plant regularly for changes.
        • Remove dead or damaged leaves.
        • Maintain good garden or farm hygiene.
        • Keep enough space between plants for good air circulation.
        """
    )

elif "Apple Scab" in best_name:

    st.warning(
        """
        🍎 Recommendation:

        • Remove and safely dispose of badly affected leaves.
        • Keep fallen leaves and plant debris away from the plant.
        • Improve air circulation around the plant.
        • Avoid unnecessary wetting of leaves.
        • Seek advice from an agricultural expert about suitable disease management.
        """
    )

elif "Black Rot" in best_name:

    st.warning(
        """
        🌿 Recommendation:

        • Remove affected plant material.
        • Keep the area around the plant clean.
        • Improve air circulation.
        • Monitor nearby plants for similar symptoms.
        • Seek professional agricultural advice if the disease spreads.
        """
    )

elif "Leaf Blight" in best_name:

    st.warning(
        """
        🌿 Recommendation:

        • Remove severely affected leaves.
        • Keep infected plant material away from healthy plants.
        • Avoid prolonged leaf wetness.
        • Improve air circulation around the plant.
        • Consult an agricultural expert for appropriate treatment.
        """
    )

elif "Early Blight" in best_name:

    st.warning(
        """
        🍅 Recommendation:

        • Remove severely affected leaves.
        • Keep the soil and area around plants clean.
        • Practice crop rotation where possible.
        • Avoid unnecessary watering of leaves.
        • Monitor the plant regularly for worsening symptoms.
        """
    )

elif "Late Blight" in best_name:

    st.warning(
        """
        🍅 Recommendation:

        • Remove severely affected plant material.
        • Separate severely infected plants when practical.
        • Improve air circulation.
        • Avoid prolonged leaf wetness.
        • Seek agricultural advice quickly if symptoms spread rapidly.
        """
    )

elif "Bacterial Spot" in best_name:

    st.warning(
        """
        🌱 Recommendation:

        • Remove severely affected leaves.
        • Avoid handling plants when they are wet.
        • Keep gardening tools clean.
        • Use healthy planting material.
        • Seek professional agricultural advice for severe infections.
        """
    )

elif "Powdery Mildew" in best_name:

    st.warning(
        """
        🌿 Recommendation:

        • Improve air circulation around plants.
        • Remove badly affected leaves.
        • Avoid overcrowding plants.
        • Monitor nearby plants for symptoms.
        • Seek agricultural advice about suitable disease management.
        """
    )

else:

    st.info(
        """
        🌱 General Recommendation:

        • Monitor the plant regularly.
        • Remove severely affected leaves if appropriate.
        • Keep the growing area clean.
        • Maintain good air circulation.
        • Avoid unnecessary leaf wetness.
        • If symptoms continue or worsen, consult a qualified agricultural expert.
        """
    )

# =========================================================
# IMPORTANT WARNING
# =========================================================

if best_confidence < 0.40:

    st.error(
        """
        ⚠️ IMPORTANT:

        The AI confidence is low. The recommendation above
        should not be treated as a confirmed diagnosis.

        Take another clear photo of the leaf or consult a
        qualified agricultural expert before taking action.
        """
    )

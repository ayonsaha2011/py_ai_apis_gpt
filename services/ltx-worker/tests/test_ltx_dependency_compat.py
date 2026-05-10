from __future__ import annotations

import transformers


def test_ltx_gemma_encoder_supports_installed_transformers() -> None:
    from ltx_core.text_encoders.gemma.encoders.encoder_configurator import (
        GemmaTextEncoderConfigurator,
        create_and_populate,
    )

    assert int(transformers.__version__.split(".", 1)[0]) == 4

    encoder = GemmaTextEncoderConfigurator.from_config({})
    create_and_populate(encoder)

    vision_tower = encoder.model.model.vision_tower
    assert hasattr(vision_tower, "vision_model")
    assert hasattr(vision_tower.vision_model.embeddings, "position_ids")

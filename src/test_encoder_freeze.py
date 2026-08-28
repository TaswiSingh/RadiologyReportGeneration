from models.visual_encoder import VisualEncoder


def main():

    encoder = VisualEncoder(
        d_model=512,
        pretrained=True,
        freeze_backbone=True
    )

    backbone_params = list(
        encoder.backbone.parameters()
    )

    trainable_backbone = sum(
        p.requires_grad
        for p in backbone_params
    )

    total_backbone = len(backbone_params)

    projection_trainable = all(
        p.requires_grad
        for p in encoder.projection.parameters()
    )

    norm_trainable = all(
        p.requires_grad
        for p in encoder.norm.parameters()
    )

    print("Total backbone parameter tensors:", total_backbone)
    print("Trainable backbone parameter tensors:", trainable_backbone)

    print(
        "Projection trainable:",
        projection_trainable
    )

    print(
        "LayerNorm trainable:",
        norm_trainable
    )

    assert trainable_backbone == 0
    assert projection_trainable
    assert norm_trainable

    print("\nEncoder freezing test PASSED!")


if __name__ == "__main__":
    main()
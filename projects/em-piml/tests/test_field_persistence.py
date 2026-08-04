from __future__ import annotations

from em_piml.model import CavityPINN
from em_piml.train import save_field_grid_artifact
from mx_viz import io as viz_io


def test_save_field_grid_artifact_roundtrips(tmp_path):
    # Untrained model -- this test is about the persistence wiring, not model accuracy, so it
    # stays fast (no @pytest.mark.slow, which is reserved for tests that actually train).
    model = CavityPINN(hidden=32, num_layers=3)
    path = tmp_path / "field.npz"

    save_field_grid_artifact(model, str(path), n_x=5, n_t=5)

    data = viz_io.load_field_artifact(path)
    viz_io.validate_field_artifact(data)
    assert data["grid_x"].shape == (5, 5)
    assert data["x"].shape == (5,)
    assert data["t"].shape == (5,)

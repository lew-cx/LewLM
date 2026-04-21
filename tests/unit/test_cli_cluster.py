from __future__ import annotations

import json
from pathlib import Path

from pydantic import SecretStr

from lewlm.cli.main import main
from lewlm.core.bootstrap import bootstrap_services


def test_cli_cluster_status_and_issue_token_emit_json(temp_settings, tmp_path: Path, capsys) -> None:
    coordinator_settings = temp_settings.with_updates(
        data_dir=tmp_path / "coordinator",
        cluster_role="coordinator",
        cluster_name="cli-cluster",
        cluster_node_name="cli-coordinator",
        cluster_public_base_url="http://cli-coordinator",
        cluster_enrollment_secret=SecretStr("cli-secret"),
    )
    services = bootstrap_services(coordinator_settings)
    try:
        status_code = main(["cluster", "status", "--json"], settings=coordinator_settings, services=services)
        status_payload = json.loads(capsys.readouterr().out)
        assert status_code == 0
        assert status_payload["role"] == "coordinator"
        assert status_payload["cluster_name"] == "cli-cluster"

        token_code = main(
            ["cluster", "issue-token", "--worker-name", "worker-a", "--json"],
            settings=coordinator_settings,
            services=services,
        )
        token_payload = json.loads(capsys.readouterr().out)
        assert token_code == 0
        assert token_payload["cluster_name"] == "cli-cluster"
        assert token_payload["worker_name"] == "worker-a"
        assert token_payload["token"]
    finally:
        services.close()

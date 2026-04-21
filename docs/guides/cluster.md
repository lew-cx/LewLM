# Cluster workflows

LewLM includes an **experimental** cluster surface for coordinator/worker enrollment and distributed planning.

## Roles

Cluster roles are configured through `LEWLM_CLUSTER_ROLE`:

- `standalone`
- `coordinator`
- `worker`

## CLI workflow

```bash
lewlm cluster status
lewlm cluster issue-token
lewlm cluster join
lewlm cluster heartbeat
lewlm cluster plan --model <model-id>
lewlm cluster benchmark --model <model-id>
```

## HTTP workflow

- `GET /v1/cluster/status`
- `POST /v1/cluster/tokens`
- `POST /v1/cluster/workers/enroll`
- `POST /v1/cluster/workers/heartbeat`
- `POST /v1/cluster/plans`
- `POST /v1/cluster/worker/pipeline-stage`
- `GET /v1/cluster/stats`

## What this surface is for

The current implementation is best understood as:

- enrollment and heartbeat coordination
- distributed execution planning
- experimental pipeline-stage orchestration
- benchmark and diagnostic coverage for distributed proofs

It should not be treated as a production-ready multi-host serving platform yet.

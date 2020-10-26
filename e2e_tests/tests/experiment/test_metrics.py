import json
import os
import pytest
import time

from determined_common import api
from tests import config as conf
from tests import experiment as exp


@pytest.mark.e2e_cpu  # type: ignore
@pytest.mark.timeout(240)
def test_streaming_metric_names() -> None:
    experiment_id = exp.create_experiment(conf.fixtures_path("no_op/single-medium-train-step.yaml"), conf.fixtures_path("no_op"), None)
    exp.wait_for_experiment_state(experiment_id, "COMPLETED")
    # The above steps are unnecessary but if they're skipped, api.get() fails to authenticate()?

    experiment_id = exp.create_experiment(
        conf.fixtures_path("no_op/single-medium-train-step.yaml"),
        conf.fixtures_path("no_op"))
    
    # This request starts immediately after the experiment, and will return when it completes
    # If we timeout, it means it failed 
    response = api.get(conf.make_master_url(), "api/v1/experiments/{}/metrics-stream/metric-names".format(experiment_id))
    results = [message['result'] for message in map(json.loads, response.text.splitlines())]

    # First let's verify an empty response was sent back before any real work was done
    assert(results[0]['searcher'] == 'validation_error')
    assert(results[0]['training'] == [])
    assert(results[0]['validation'] == [])

    # Then we verify that all expected responses are eventually received exactly once
    accumulated_training = set()
    accumulated_validation = set()
    for i in range(1, len(results)):
        for training in results[i]['training']:
            assert(not training in accumulated_training)
            accumulated_training.add(training)
        for validation in results[i]['validation']:
            assert(not validation in accumulated_validation)
            accumulated_validation.add(validation)
    assert(accumulated_training == set(['loss']))
    assert(accumulated_validation == set(['validation_error']))


@pytest.mark.e2e_cpu  # type: ignore
@pytest.mark.timeout(90)
def test_streaming_metric_batches() -> None:
    experiment_id = exp.create_experiment(
        conf.fixtures_path("no_op/single-medium-train-step.yaml"),
        conf.fixtures_path("no_op"))
    
    # This request 
    response = api.get(conf.make_master_url(),
        "api/v1/experiments/{}/metrics-stream/batches".format(experiment_id),
        params={"training_metric": "loss"})
    results = [message['result'] for message in map(json.loads, response.text.splitlines())]

    # First let's verify an empty response was sent back before any real work was done
    assert(results[0]['batches'] == [])

    # Then we verify that all expected responses are eventually received exactly once
    accumulated = set()
    for i in range(1, len(results)):
        for batch in results[i]['batches']:
            assert(not batch in accumulated)
            accumulated.add(batch)
    assert(accumulated == set([100, 200, 300, 400, 500]))

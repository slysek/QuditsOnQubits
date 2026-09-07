import hashlib
import json
import zipfile

import pytest
import reproduce


def bundle(tmp_path, name='jobs/test/counts.json', data=b'{}'):
    archive = tmp_path / 'data.zip'
    with zipfile.ZipFile(archive, 'w') as output:
        output.writestr(name, data)
    manifest = {'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                'files': {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}}}
    return archive, manifest


def test_archive_verification_detects_changed_data(tmp_path):
    archive, manifest = bundle(tmp_path)
    reproduce.verify_archive(archive, manifest)
    manifest['files']['jobs/test/counts.json']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='checksum'):
        reproduce.verify_archive(archive, manifest)


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'C:/absolute', 'a/../../escape', 'a\\escape'])
def test_unsafe_archive_paths_rejected(tmp_path, name):
    archive, manifest = bundle(tmp_path, name)
    with pytest.raises(ValueError, match='path'):
        reproduce.verify_archive(archive, manifest)


def test_counts_must_have_exact_requested_shots():
    receipt = {'status': 'completed', 'shots': 5000, 'circuit_count': 1, 'mapping': [{}]}
    reproduce.validate_counts(receipt, [{'00': 3000, '01': 2000}])
    with pytest.raises(ValueError, match='shots'):
        reproduce.validate_counts(receipt, [{'00': 4999}])
    with pytest.raises(ValueError, match='counts'):
        reproduce.validate_counts(receipt, [{'00': 5001, '01': -1}])


def test_compare_checks_missing_or_changed_estimates():
    expected = [{'state': 'two', 'name': 'baseline', 'variant': 'optimal',
                 'raw_conditional': {'value': 4.5}}]
    reproduce.compare_estimates(expected, expected, keys=['raw_conditional'])
    with pytest.raises(ValueError, match='groups'):
        reproduce.compare_estimates([], expected, keys=['raw_conditional'])
    changed = json.loads(json.dumps(expected))
    changed[0]['raw_conditional']['value'] = 4.6
    with pytest.raises(ValueError, match='estimate'):
        reproduce.compare_estimates(changed, expected, keys=['raw_conditional'])


def test_full_archive_replays_without_network_and_does_not_overwrite(tmp_path, monkeypatch):
    import socket
    def denied(*args, **kwargs):
        raise AssertionError('Offline replay attempted network access')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)
    output = tmp_path / 'replayed'
    reproduce.replay(output)
    summary = json.loads((output / 'verification.json').read_text())
    assert summary['verified_completed_jobs'] == 43
    assert summary['aggregate_rows'] == 240
    assert summary['published_primary_values_match']
    assert not summary['bootstrap_recomputed']
    sentinel = output / 'user-file.txt'
    sentinel.write_text('preserve')
    with pytest.raises(FileExistsError):
        reproduce.replay(output)
    assert sentinel.read_text() == 'preserve'


@pytest.mark.parametrize('field,value', [('invalid_fraction', .2), ('total_shots', 9), ('shots_per_setting', [4, 5])])
def test_compare_rejects_changed_postselection_metadata(field, value):
    expected = [{'state': 'two', 'name': 'base', 'variant': 'standard',
                 'raw_conditional': {'value': 4.5}, 'invalid_fraction': .1,
                 'total_shots': 10, 'shots_per_setting': [5, 5]}]
    actual = json.loads(json.dumps(expected))
    actual[0][field] = value
    with pytest.raises(ValueError, match='metadata'):
        reproduce.compare_estimates(actual, expected, keys=['raw_conditional'])


def test_postselection_aggregation_weights_by_shots(tmp_path):
    import numpy as np
    job = {'series': 'original', 'backend': 'garnet'}
    rows = []
    for count, invalid in [(100, .1), (300, .5)]:
        row = {'state': 'two', 'name': 'canonical_ez', 'variant': 'standard', 'factor': 1,
               'total_shots': count, 'invalid_fraction': invalid, 'job_id': str(count)}
        row.update({key: {'value': .5} for key in reproduce.KEYS})
        rows.append((job, row, np.zeros((10, 4))))
    result = reproduce.aggregate(rows, tmp_path, False)
    assert result[0]['invalid_fraction'] == pytest.approx(.4)

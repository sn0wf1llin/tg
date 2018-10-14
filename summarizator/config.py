"""Declare paths."""
from os import path, makedirs
from psettings import DEFAULT_PROJECT_PATH


path_base = DEFAULT_PROJECT_PATH + 'tg/summarizator'
path_data = path_base + '/data'
path_models = path_base + '/models'
path_logs = path_models + '/logs'
path_visual_reports = path_base + '/reports'

# verify output path exists otherwise make it
for p in [path_data, path_models, path_logs, path_visual_reports]:
    if not path.exists(p):
        makedirs(p)

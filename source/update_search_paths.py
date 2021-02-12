import os.path
import sys

_source_dir = os.path.dirname(__file__)
sys.path.append(_source_dir)
sys.path.append(os.path.join(_source_dir, 'cmd_utils'))
sys.path.append(os.path.join(_source_dir, 'bayesian'))
sys.path.append(os.path.join(os.path.dirname(_source_dir), 'data'))

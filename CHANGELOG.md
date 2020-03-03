# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/).

Please keep the latest version in sync with `version.py`.

## [1.1.0] - 2020-03-03
### Added
- Log-loss metric to `demo_ripley.py`

### Changed
- Extended to multiclass classification

### Fixed
- Bug in `feature_importance()`: Now normalizes hyperplane normal
  vector to avoid stiffness leaking into feature importance

## [1.0.0] - 2020-02-28
### Added
- Initial public release

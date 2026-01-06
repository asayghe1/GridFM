# Contributing to GridFM

We welcome contributions to GridFM! This document provides guidelines for contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/GridFM.git
   cd GridFM
   ```
3. **Create a virtual environment**:
   ```bash
   conda create -n gridfm-dev python=3.10
   conda activate gridfm-dev
   ```
4. **Install development dependencies**:
   ```bash
   pip install -e ".[dev,all]"
   ```

## Development Workflow

### Branching Strategy

- `main`: Stable release branch
- `develop`: Integration branch for features
- `feature/*`: Feature branches
- `bugfix/*`: Bug fix branches
- `release/*`: Release preparation branches

### Creating a Feature Branch

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

### Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Before committing, run:
```bash
black gridfm tests scripts
isort gridfm tests scripts
flake8 gridfm tests scripts
mypy gridfm
```

### Testing

We use pytest for testing. Run tests with:
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=gridfm tests/

# Run specific test file
pytest tests/test_freqmixer.py -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use fixtures for common setup
- Aim for high coverage of critical paths

Example:
```python
import pytest
import torch
from gridfm.layers.freqmixer import FreqMixer

def test_freqmixer_output_shape():
    freqmixer = FreqMixer(input_dim=64, hidden_dim=128)
    x = torch.randn(2, 288, 64)
    output, _ = freqmixer(x)
    assert output.shape == (2, 128)
```

## Pull Request Process

1. **Update documentation** for any new features
2. **Add tests** for new functionality
3. **Ensure all tests pass** locally
4. **Update CHANGELOG.md** with your changes
5. **Submit a pull request** to the `develop` branch

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests passing

## Checklist
- [ ] Code follows project style guidelines
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Publishing private information
- Other unprofessional conduct

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. **Environment details**: OS, Python version, PyTorch version
2. **Steps to reproduce**: Minimal code example
3. **Expected behavior**: What you expected
4. **Actual behavior**: What actually happened
5. **Error messages**: Full traceback if applicable

### Feature Requests

For feature requests, please include:

1. **Use case**: Why this feature is needed
2. **Proposed solution**: How you envision it working
3. **Alternatives considered**: Other approaches you've thought of

## License

By contributing to GridFM, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue for any questions not covered here.

Thank you for contributing to GridFM! 🎉

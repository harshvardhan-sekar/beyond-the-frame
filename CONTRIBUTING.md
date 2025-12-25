# Contributing to COMICS Cloze VLM

Thank you for your interest in contributing to this project! 

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists in the [Issues](https://github.com/YOUR_USERNAME/comics-cloze-vlm/issues) tab
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Environment details (OS, Python version, GPU)

### Code Contributions

1. **Fork** the repository
2. **Clone** your fork locally
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** following the code style guidelines
5. **Test** your changes thoroughly
6. **Commit** with clear messages:
   ```bash
   git commit -m "Add: brief description of changes"
   ```
7. **Push** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
8. Open a **Pull Request** with a clear description

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Add comments for complex logic

### Notebook Guidelines

- Clear markdown cells explaining each section
- Remove unnecessary outputs before committing
- Include expected runtimes for long-running cells
- Document any required files or dependencies

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/comics-cloze-vlm.git
cd comics-cloze-vlm

# Create development environment
conda env create -f envs/environment.yml
conda activate comics-vlm

# Install development dependencies
pip install black flake8 pytest
```

## Questions?

Feel free to open an issue for questions or reach out to the maintainers.

Thank you for contributing! 🎉

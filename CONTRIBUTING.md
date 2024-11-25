# Contributing to AI Dashboard Builder

Thank you for your interest in contributing to AI Dashboard Builder! We welcome contributions from the community and are excited to have you aboard.

## How to Contribute

### 1. Setting Up Development Environment

1. Fork the repository
2. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/ai-dashboard-builder.git
```
3. Set up the development environment:
```bash
PYTHONPATH=$PYTHONPATH:./src python src/app.py
```

### 2. Making Changes

1. Create a new branch for your feature/fix:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes following our coding standards:
   - Keep code simple and readable
   - Add comments for complex logic
   - Try to follow PEP 8 style guidelines
   - Maintain the existing simple code structure

3. Test your changes:
   - Ensure all existing tests pass
   - Add new tests for new functionality
   - Test with different LLM providers (Ollama, OpenAI, etc.)

### 3. Submitting Changes

1. Commit your changes with clear, descriptive messages:
```bash
git commit -m "Add: brief description of changes"
```

2. Push to your fork:
```bash
git push origin feature/your-feature-name
```

3. Create a Pull Request (PR):
   - Provide a clear description of the changes. PRs should be focused on a single feature or bug fix
   - Link any related issues
   - Include screenshots for UI changes
   - List any new dependencies

## Code Style Guidelines

- Use meaningful variable and function names
- Keep functions focused and single-purpose
- Document new functions and classes
- Use type hints where applicable
- Format code using a consistent style

## Project Structure

```
ai-dashboard-builder/
├── src/
│   ├── app.py           # Main application entry point
│   ├── llm/             # LLM integration modules
│   └── ...
├── docker/              # Individual container Dockerfiles
└── ...
```

## Testing

- Write unit tests for new functionality
- Ensure tests are deterministic
- Test with different datasets
- Verify compatibility with supported LLM providers

## Documentation

When adding new features, please update:
- README.md for user-facing changes
- Code comments for technical details
- API documentation if applicable

## Need Help?

- Open an issue for questions
- Join our discussions
- Review existing PRs and issues

## License

By contributing to AI Dashboard Builder, you agree that your contributions will be licensed under the MIT License. 
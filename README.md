# Edumate: AI-Powered Educational Platform

Edumate is an intelligent educational platform designed to enhance teaching and learning experiences through AI-powered tools and analytics.

## Features

- **Test Creator**: Generate customized quizzes and assessments
  - Quiz Generator: Create quizzes with multiple question types
  - Lesson Plan Creator: Build comprehensive lesson plans
  - Rubric Builder: Design detailed grading rubrics

- **Student Progress Tracking**: Monitor student performance and identify areas for improvement

- **AI Tutor**: Provide personalized learning experiences through AI-powered tutoring

- **Educational Resources**: Access and manage curated educational materials

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/edumate.git
cd edumate

# Install the package
pip install -e .
```

## Usage

```bash
# Run the Streamlit application
streamlit run edumate/app.py

# Or use the entry point script
edumate
```

## Project Structure

```
edumate/
├── app.py                 # Main application entry point
├── ai/                    # AI components and models
├── pages/                 # Streamlit page components
│   └── test_creator.py    # Test creation interface
├── services/              # Service layer components
├── utils/                 # Utility functions and helpers
│   └── quiz_generator.py  # Quiz generation utilities
├── models/                # Data models
└── views/                 # View components
```

## Development

This project is under active development. The current version is migrating from a monolithic application structure to a modular, maintainable architecture.

### Migration Progress

- [x] Create modular structure
- [x] Extract test creator functionality
- [ ] Extract student progress tracking
- [ ] Extract AI tutor functionality
- [ ] Extract resources management
- [ ] Complete test coverage

## License

[MIT License](LICENSE)

## Contributors

- Edumate Team 
# Waste Classifier Sustainability App

## Overview

This Waste Classifier Sustainability App uses a trained machine learning model to classify images of various waste items and provides information about their carbon footprint. The application is built using Streamlit for the user interface, Keras for the machine learning model, and OpenAI's GPT-3 for generating detailed information on the carbon footprint.

## Features

- **Image Classification**: Classifies waste items into predefined categories such as motherboards, batteries, mobiles, and washing machines.
- **Carbon Footprint Information**: Provides an estimate of the carbon emissions associated with each type of waste item.
- **Interactive UI**: User-friendly interface powered by Streamlit.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/SimpleCyber/E-Waste-Clasifier.git
cd E-Waste-Clasifier
```

2. Install the required dependencies:

```bash
pip install streamlit keras pillow numpy python-dotenv openai
```

3. Ensure you have your OpenAI API key stored in a `.env` file in the project directory:

```
OPENAI_API_KEY=your_openai_api_key
```

4. Run the application:

```bash
streamlit run app.py
```

## Usage

1. Upload an image of a waste item (jpg, png, jpeg).
2. Click the "Classify" button to classify the image.
3. View the classification result and information about the carbon footprint of the classified item.

## Contribution

Contributions are welcome! Feel free to submit a pull request or open an issue for any feature requests or bugs.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Thank you for using the Waste Classifier Sustainability App!

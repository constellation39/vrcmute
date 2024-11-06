# VRCMute

VRCMute is a VRChat mute assistant that uses speech-to-text conversion (powered by DashScope API) to help manage your microphone in VRChat based on speech content.

## Features

- Real-time speech-to-text conversion using DashScope API
- Automatic mute functionality for VRChat
- Content-based muting customization

## Prerequisites

- Windows 10/11
- Python 3.12 or higher (release not required.)
- A valid DashScope API key for paraformer-realtime integration

## Installation

### From Release

1. Download the latest release from [Releases](https://github.com/Constellation39/vrcmute/releases)
2. Extract the ZIP file to your preferred location
3. Set up your environment variables (see Configuration section)

### Building from Source

```bash
git clone https://github.com/Constellation39/vrcmute.git
cd vrcmute

python -m venv venv
.\venv\Scripts\activate

pip install .
```

## Configuration

### Setting up DashScope API Key

1. Visit [DashScope Console](https://dashscope.console.aliyun.com/)
2. Register or log in to your account
3. Navigate to the API Keys section
4. Create a new API key if you don't have one
5. Set up your API key as an environment variable:

```powershell
# Windows (PowerShell)
$env:DASHSCOPE_API_KEY='your_api_key_here'

# Or set it permanently through Windows Settings:
# 1. Open System Properties -> Advanced -> Environment Variables
# 2. Add a new User variable:
#    Name: DASHSCOPE_API_KEY
#    Value: your_api_key_here
```

## Usage

1. Start VRChat
2. Run VRCMute:
   ```bash
   # If installed from release:
   ./vrcmute.exe

   # If running from source:
   python -m src.__main__
   ```
3. The program will automatically detect your microphone and start monitoring speech

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run linting
ruff check .
ruff format .

# Run type checking
mypy src
```

## Building from Source

```bash
# Install PyInstaller
pip install pyinstaller

# Build the executable
pyinstaller vrcmute.spec
```

The built executable will be available in the `dist` directory.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) for speech-to-text functionality
- [DashScope](https://dashscope.aliyun.com/) for providing the paraformer-realtime model
- VRChat community for inspiration and support

## Troubleshooting

### Common Issues

1. **CUDA Issues**
   - Ensure CUDA Toolkit 12.1 is properly installed
   - Check your GPU drivers are up to date
   - Verify CUDA paths are correctly set in environment variables

2. **DashScope API Issues**
   - Verify your API key is correctly set
   - Check your account has sufficient credits
   - Ensure your network can access DashScope services

3. **VRChat Connection Issues**
   - Verify VRChat is running
   - Check if OSC is enabled in VRChat settings
   - Ensure no firewall is blocking the connection

For more issues, please check the [Issues](https://github.com/Constellation39/vrcmute/issues) page.
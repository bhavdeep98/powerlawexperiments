# Setting Up Your API Key

## Quick Setup

1. **Copy the example file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and paste your API key**:
   ```bash
   # Open .env in your editor
   nano .env
   # or
   code .env
   ```

3. **Add your key**:
   ```
   OPENAI_API_KEY=sk-proj-your-actual-key-here
   ```

4. **Save the file** - That's it! The code will automatically load it.

## Alternative: Environment Variable

If you prefer not to use a `.env` file, you can export it:

```bash
export OPENAI_API_KEY=sk-proj-your-actual-key-here
```

## Verify Setup

Run the experiment to verify:
```bash
python3 agentic_scaling_experiment.py --pilot
```

If you see the experiment starting (not an API key error), you're all set!

## Security Note

- ✅ `.env` is in `.gitignore` - your key won't be committed
- ✅ `.env.example` is safe to commit (no real keys)
- ⚠️ Never commit your actual `.env` file

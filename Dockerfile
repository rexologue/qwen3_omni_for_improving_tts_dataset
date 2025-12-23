FROM qwenllm/qwen3-omni:latest

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 1. Install dependencies as ROOT (Global install)
# Note: Ensure your local file is named 'requiremetns.txt' or 'requirements.txt' to match source
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 2. BAKE THE MODELS (Download them now, as root)
# This executes the same load logic your script uses, forcing the download into the image.
RUN python3 -c "from ruaccent import RUAccent; accentizer = RUAccent(); accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True, tiny_mode=False)"

# 3. FIX PERMISSIONS (The Critical Step)
# RuAccent insists on writing to its install dir. We allow ANY user (777) to write there.
# This fixes the "PermissionError: .../ruaccent/.cache"
RUN chmod -R 777 /usr/local/lib/python3.10/dist-packages/ruaccent

# 4. Create User & App Setup
RUN groupadd -r omni && useradd -r -g omni -m -d /home/omni omni
WORKDIR /home/omni/app

# Copy application code
COPY . .

# Fix App permissions for the runtime user
RUN chown -R omni:omni /home/omni/app && chmod -R 777 /home/omni

# Switch to non-root user
USER omni

CMD ["python3", "tag_dataset.py", "--help"]
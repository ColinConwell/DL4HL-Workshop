"""Reject incomplete executed HTML and notebook/source drift."""
from pathlib import Path
import ast
import hashlib
import re
import nbformat

ROOT = Path(__file__).absolute().parents[1]
source = (ROOT / "DL4HL-Tutorial.qmd").read_text()
page = (ROOT / "DL4HL-Tutorial.html").read_text()

labels = re.findall(r"^#\| label: (.+)$", source, re.M)
assert page.count('class="cell"') >= len(labels), (page.count('class="cell"'), len(labels))
assert page.count('<div class="ml4hl-plot">') >= 8, f"Expected >= 8 themed figures, got {page.count('<div class=\"ml4hl-plot\">')}"
assert "data:image/png;base64," in page and "data:image/jpeg;base64," in page
assert "Traceback (most recent call last)" not in page

nb = nbformat.read(ROOT / "DL4HL-Tutorial.ipynb", 4)
assert sum(c.cell_type == "code" for c in nb.cells) == len(re.findall(r"^```\{python\}", source, re.M)) + 1
assert "urlopen(_helper_url" in nb.cells[1].source and "DL4HL_STANDALONE" in nb.cells[1].source
assert "hashlib.sha256(_helper_zip)" in nb.cells[1].source
assert max(map(len, nb.cells[1].source.splitlines())) < 200, "Oversized notebook setup line"

setup = ast.parse(nb.cells[1].source)
helper_hash = next(
    node.value.value
    for node in setup.body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "_helper_sha256" for target in node.targets)
)
assert re.fullmatch(r"[a-f0-9]{64}", helper_hash)
helper = ROOT / "assets" / "helpers" / f"dl4hl-{helper_hash}.zip"
assert hashlib.sha256(helper.read_bytes()).hexdigest() == helper_hash, "Missing or altered helper archive"
print(f"PASS: {len(labels)} labeled source cells, {page.count('<div class=\"ml4hl-plot\">')} themed figures, logo, and generated notebook setup")

blocks = re.findall(r"```\{python\}\n(.*?)\n```", source, re.S)
figure_blocks = [block for block in blocks if "plt.subplots(" in block]
assert all("#| code-fold: true" in block for block in figure_blocks), "Unfolded plotting cell"
figure_cells = [cell for cell in nb.cells if cell.cell_type == "code" and "plt.subplots(" in cell.source]
assert len(figure_cells) == len(figure_blocks)
assert all(cell.metadata.get("cellView") == "form" for cell in figure_cells), "Colab plot folding lost"

takeaways = re.findall(r'title="Takeaway #\d+"}\n(.*?)\n:::', source, re.S)
closing = source.split("# Wrap-Up + Takeaways\n")[1].split("## The Whole Toolkit")[0]
assert len(takeaways) == 10, f"Expected 10 takeaways, got {len(takeaways)}"
assert all(f"{i}. {text}" in closing for i, text in enumerate(takeaways, 1)), "Closing takeaways drifted"

stack = []
for line in source.splitlines():
    if re.match(r"^:::+\s*\{", line):
        stack.append(line)
    elif re.match(r"^:::+\s*$", line):
        assert stack, "Closing a Markdown container that is not open"
        stack.pop()
assert not stack, f"Unclosed Markdown containers: {stack}"

widget_blocks = [block for block in blocks if re.search(r"^show\(", block, re.M)]
assert len(widget_blocks) == 12, f"Expected 12 interactive widgets, got {len(widget_blocks)}"
assert all("#| echo: false" in block or "#| code-fold: true" in block for block in widget_blocks), "Interactive figure code is not hidden or folded"
assert 'class="conwell-paper-fig"' in page, "Missing Conwell et al. (2025) Figure 12A/12B embed"
assert "# Glossary {.unnumbered}" in source and "# Data Sources {.unnumbered}" in source, "Expected unnumbered primary Glossary and Data Sources headers"
assert "DeepJuice" not in page and "Sim-Acela" not in page, "Unreleased package reference remains"
print(f"PASS: {len(figure_blocks)} folded plot cells, {len(widget_blocks)} interactive widgets, Conwell (2025) figures, unnumbered TOC headers, and 10 synchronized takeaways")

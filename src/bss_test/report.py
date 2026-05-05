"""
Experiment report generation.

Generates self-contained HTML and Markdown reports from experiment results,
including embedded figures, metrics tables, and method comparisons.

Usage:
    from bss_test.report import ExperimentReport

    report = ExperimentReport("BSS Method Comparison", output_dir="outputs/bss")
    report.add_text("Setup", "CWRU inner race fault data...")
    report.add_figure(fig, "Envelope Spectrum Comparison")
    report.add_metrics_table(results, "Performance Metrics")
    report.to_html()
    report.to_markdown()
"""

import base64
import html as html_module
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from bss_test.utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentReport:
    """Experiment report generator supporting HTML and Markdown output."""

    def __init__(self, title: str, output_dir: str):
        self.title = title
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sections: List[Dict] = []
        self.metadata: Dict[str, str] = {
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def set_metadata(self, key: str, value: str):
        """Add metadata (shown in report header)."""
        self.metadata[key] = value

    def add_text(self, heading: str, content: str):
        """Add a text section."""
        self.sections.append({"type": "text", "heading": heading, "content": content})

    def add_figure(self, fig, caption: str, filename: str = None):
        """Add a matplotlib figure.

        Parameters
        ----------
        fig : matplotlib Figure
        caption : str
        filename : str or None
            If None, auto-generates from caption.
        """
        if filename is None:
            filename = caption.lower().replace(" ", "_").replace("/", "_")[:40] + ".png"
        self.sections.append({"type": "figure", "fig": fig, "caption": caption, "filename": filename})

    def add_table(self, headers: List[str], rows: List[List], caption: str = ""):
        """Add a table."""
        self.sections.append({"type": "table", "headers": headers, "rows": rows, "caption": caption})

    def add_metrics_table(self, results: List[Dict], caption: str = ""):
        """Add a metrics table from classifier results.

        Parameters
        ----------
        results : list of dict
            Each dict should have 'method', 'accuracy', 'f1_macro', and optionally 'time'.
        caption : str
        """
        headers = ["Method", "Accuracy", "F1-Macro"]
        has_time = any("time" in r for r in results)
        if has_time:
            headers.append("Time (s)")

        rows = []
        for r in results:
            row = [
                r.get("method", "").upper(),
                f"{r.get('accuracy', 0):.4f}",
                f"{r.get('f1_macro', 0):.4f}",
            ]
            if has_time:
                row.append(f"{r.get('time', 0):.2f}")
            rows.append(row)

        self.sections.append({"type": "table", "headers": headers, "rows": rows, "caption": caption})

    def to_html(self, filepath: str = None) -> str:
        """Generate self-contained HTML report.

        Figures are embedded as base64 data URIs. CSS is inline.
        """
        if filepath is None:
            filepath = str(self.output_dir / "report.html")

        html_parts = [self._html_header()]

        for section in self.sections:
            if section["type"] == "text":
                html_parts.append(self._html_text(section))
            elif section["type"] == "figure":
                html_parts.append(self._html_figure(section))
            elif section["type"] == "table":
                html_parts.append(self._html_table(section))

        html_parts.append(self._html_footer())

        html_content = "\n".join(html_parts)

        Path(filepath).write_text(html_content, encoding="utf-8")
        logger.info(f"HTML report saved: {filepath}")
        return html_content

    def to_markdown(self, filepath: str = None) -> str:
        """Generate Markdown report.

        Figures are saved as PNG files in the output directory.
        """
        if filepath is None:
            filepath = str(self.output_dir / "report.md")

        md_parts = [self._md_header()]

        for section in self.sections:
            if section["type"] == "text":
                md_parts.append(self._md_text(section))
            elif section["type"] == "figure":
                md_parts.append(self._md_figure(section))
            elif section["type"] == "table":
                md_parts.append(self._md_table(section))

        md_content = "\n".join(md_parts)

        Path(filepath).write_text(md_content, encoding="utf-8")
        logger.info(f"Markdown report saved: {filepath}")
        return md_content

    # ---- HTML helpers ----

    def _html_header(self) -> str:
        meta_lines = "".join(
            f"<tr><td><strong>{html_module.escape(str(k))}</strong></td>"
            f"<td>{html_module.escape(str(v))}</td></tr>\n"
            for k, v in self.metadata.items()
        )
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{self.title}</title>
<style>
body {{ font-family: 'Times New Roman', serif; max-width: 1000px; margin: 0 auto; padding: 20px;
       background: #fff; color: #333; line-height: 1.6; }}
h1 {{ font-size: 1.8em; border-bottom: 2px solid #333; padding-bottom: 8px; }}
h2 {{ font-size: 1.3em; color: #1976D2; margin-top: 1.5em; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
th {{ background: #f5f5f5; font-weight: bold; }}
tr:nth-child(even) {{ background: #fafafa; }}
.meta-table {{ width: auto; margin-bottom: 1.5em; }}
.meta-table td {{ border: none; padding: 2px 10px; }}
.figure {{ text-align: center; margin: 1.5em 0; }}
.figure img {{ max-width: 100%; height: auto; }}
.figure .caption {{ font-style: italic; color: #555; margin-top: 0.5em; font-size: 0.95em; }}
.summary {{ background: #f0f7ff; padding: 12px 16px; border-radius: 4px; border-left: 4px solid #1976D2; }}
</style>
</head>
<body>
<h1>{self.title}</h1>
<table class="meta-table">
{meta_lines}
</table>
"""

    def _html_text(self, section: Dict) -> str:
        heading = html_module.escape(str(section['heading']))
        content = html_module.escape(str(section['content']))
        return f"<h2>{heading}</h2>\n<p>{content}</p>"

    def _html_figure(self, section: Dict) -> str:
        fig = section["fig"]
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        caption = html_module.escape(str(section["caption"]))
        return (
            f'<div class="figure">\n'
            f'<img src="data:image/png;base64,{b64}" alt="{caption}">\n'
            f'<div class="caption">{caption}</div>\n'
            f'</div>'
        )

    def _html_table(self, section: Dict) -> str:
        headers = section["headers"]
        rows = section["rows"]
        caption = section.get("caption", "")

        thead = "".join(f"<th>{html_module.escape(str(h))}</th>" for h in headers)
        body_rows = []
        for row in rows:
            cells = "".join(f"<td>{html_module.escape(str(c))}</td>" for c in row)
            body_rows.append(f"<tr>{cells}</tr>")
        tbody = "\n".join(body_rows)

        caption_html = f"<caption><strong>{caption}</strong></caption>" if caption else ""
        return f"<table>{caption_html}\n<thead><tr>{thead}</tr></thead>\n<tbody>\n{tbody}\n</tbody>\n</table>"

    def _html_footer(self) -> str:
        return f"\n</body>\n</html>"

    # ---- Markdown helpers ----

    def _md_header(self) -> str:
        meta_lines = "\n".join(f"- **{k}**: {v}" for k, v in self.metadata.items())
        return f"# {self.title}\n\n{meta_lines}\n"

    def _md_text(self, section: Dict) -> str:
        return f"\n## {section['heading']}\n\n{section['content']}\n"

    def _md_figure(self, section: Dict) -> str:
        fig = section["fig"]
        filename = section["filename"]
        filepath = self.output_dir / filename
        fig.savefig(str(filepath), dpi=200, bbox_inches="tight")
        return f"\n![{section['caption']}]({filename})\n*{section['caption']}*\n"

    def _md_table(self, section: Dict) -> str:
        headers = section["headers"]
        rows = section["rows"]
        caption = section.get("caption", "")

        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        data_lines = []
        for row in rows:
            data_lines.append("| " + " | ".join(str(c) for c in row) + " |")

        result = ""
        if caption:
            result += f"\n**{caption}**\n\n"
        result += header_line + "\n" + sep_line + "\n" + "\n".join(data_lines) + "\n"
        return result

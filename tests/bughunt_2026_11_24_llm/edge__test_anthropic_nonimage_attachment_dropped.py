"""A non-image binary attachment (e.g. a PDF) left in msg.files after
embed_attached_file_content() is silently dropped by the Anthropic provider:
process_image_file() returns None for unsupported extensions and
llm_anthropic._process_file just `continue`s, adding nothing. The model never
learns a file was attached. The OpenAI provider, for the same input, appends
"[WARNING: Model doesn't support viewing file: ...]".
"""

from gptme.llm.llm_anthropic import _prepare_messages_for_api as anth_prepare
from gptme.llm.llm_openai import _prepare_messages_for_api as oai_prepare
from gptme.message import Message


def _msgs(pdf):
    return [
        Message(role="system", content="sys"),
        Message(role="user", content="Summarize the attached report.", files=[pdf]),
    ]


def test_openai_mentions_unsupported_attachment(tmp_path):
    pdf = tmp_path / "q3-report.pdf"
    pdf.write_bytes(b"%PDF-1.7\n\xff\xfe binary")
    dicts, _ = oai_prepare(_msgs(pdf), "openai/gpt-4o", None)
    assert "q3-report.pdf" in repr(dicts)  # baseline: passes on master


def test_anthropic_mentions_unsupported_attachment(tmp_path):
    pdf = tmp_path / "q3-report.pdf"
    pdf.write_bytes(b"%PDF-1.7\n\xff\xfe binary")
    dicts, _, _ = anth_prepare(_msgs(pdf), None)
    # On master the attachment vanishes without any trace in the payload.
    assert "q3-report.pdf" in repr(dicts), dicts

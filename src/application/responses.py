from io import BytesIO
from pathlib import Path

from litestar.response import Response

__all__ = ['ZipResponse', 'LogFileResponse', 'AnsiHTMLResponse']


class ZipResponse(Response[bytes]):
    def __init__(self, name: str, root: Path, paths: list[Path]) -> None:
        import zipfile
        from datetime import datetime

        buffer = BytesIO()
        name = name.format(ts=datetime.now().strftime("%y%m%d-%H%M%S"))
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for src in paths:
                if src.exists():
                    dst = src.relative_to(root).as_posix()
                    zip_file.write(src, dst)
        buffer.seek(0)
        super().__init__(
            buffer.getvalue(),
            headers={"content-disposition": f"attachment;filename={name}"},
            media_type='application/zip',
        )


class LogFileResponse(Response[str]):
    def __init__(self, name: str, text: str) -> None:
        from datetime import datetime

        name = name.format(ts=datetime.now().strftime("%y%m%d-%H%M%S"))
        super().__init__(
            text,
            headers={"content-disposition": f"attachment;filename={name}"},
            media_type='text/plain',
        )


class AnsiHTMLResponse(Response[str]):
    def __init__(self, ansi_text: str) -> None:
        from ansi2html import Ansi2HTMLConverter

        conv = Ansi2HTMLConverter()
        html = conv.convert(ansi_text)
        super().__init__(html, media_type="text/html")

import os
from typing import Any

_DEFAUL_ENCODING = "ASCII"


class XmlWriter:
    """逐步写入简单且结构完整的 XML 文档。"""

    def __init__(
        self, filepath: str | os.PathLike[str], addDeclaration: bool = True
    ) -> None:
        """打开输出文件，并按需写入 XML 声明。"""
        self.stream = open(filepath, "wb")
        self.openTag = False
        self.current: list[str] = []
        if addDeclaration:
            self.addDeclaration()

    def addComment(self, sstr: str) -> None:
        """写入一条完整的 XML 注释。"""
        if self.openTag:
            self.stream.write(b">")
            self.openTag = False
        self.stream.write(b"\n<!-- ")
        self.stream.write(sstr.encode(_DEFAUL_ENCODING))
        self.stream.write(b" -->")

    def close(self) -> None:
        """关闭已完成的 XML 文件。"""
        assert not self.openTag
        self.stream.close()

    def addDeclaration(self) -> None:
        """写入 XML 1.0 声明。"""
        self.stream.write(b'<?xml version="1.0"?>')

    def openElement(self, tag: str) -> "XmlWriter":
        """打开元素，返回自身以支持链式调用。"""
        if self.openTag:
            self.stream.write(b">")
        st = f"\n<{tag}"
        self.stream.write(st.encode(_DEFAUL_ENCODING))
        self.openTag = True
        self.current.append(tag)
        return self

    def closeElement(self, tag: str | None = None) -> "XmlWriter":
        """关闭指定元素；省略标签时写入自闭合元素。"""
        if tag:
            assert self.current.pop() == tag
            if self.openTag:
                self.stream.write(b">")
                self.openTag = False
            st = f"\n</{tag}>"
            self.stream.write(st.encode(_DEFAUL_ENCODING))
        else:
            self.stream.write(b"/>")
            self.openTag = False
            self.current.pop()
        return self

    def addText(self, text: str) -> "XmlWriter":
        """向当前元素写入文本，返回自身以支持链式调用。"""
        if self.openTag:
            self.stream.write(b">\n")
            self.openTag = False
        self.stream.write(text.encode(_DEFAUL_ENCODING))
        return self

    def addAttributes(self, **kwargs: Any) -> "XmlWriter":
        """向尚未闭合的起始标签加入属性。"""
        assert self.openTag
        for key, value in kwargs.items():
            st = f' {key}="{value}"'
            self.stream.write(st.encode(_DEFAUL_ENCODING))
        return self

from __future__ import absolute_import

import os.path
from typing import Optional
from itertools import chain
from glob import glob

from ocrd_models.ocrd_page import OcrdPage, TextEquivType
from ocrd import Processor, OcrdPageResult

class ReplaceText(Processor):

    @property
    def executable(self):
        return 'ocrd-segment-replace-text'

    def setup(self):
        file_glob = self.parameter['file_glob']
        self.input_text_files = glob(file_glob)
        assert len(self.input_text_files), f"file_glob '{file_glob}' does not match any path names"

    def process_page_pcgts(self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None) -> OcrdPageResult:
        """Add TextEquiv anywhere below the page level from external text files named by segments.

        Open and deserialize PAGE input file. Try to find text files matching
        ``file_glob`` which have both the page ID and some segment ID in their path name.

        For every match, insert the content of the text file as first TextEquiv of that very
        segment.

        Produce a new output file by serialising the resulting hierarchy.
        """
        pcgts = input_pcgts[0]
        page = pcgts.get_Page()
        regions = page.get_AllRegions(classes=['Text'])
        lines = list(chain.from_iterable(
            [region.get_TextLine() for region in regions]))
        words = list(chain.from_iterable(
            [line.get_Word() for line in lines]))
        glyphs = list(chain.from_iterable(
            [word.get_Glyph() for word in words]))
        segids = { seg.id: seg for seg in glyphs + words + lines + regions }
        text_files = ([path for path in self.input_text_files
                       if page_id in path] or
                      [path for path in self.input_text_files
                       if input_file.ID in path])
        if not len(text_files):
            self.logger.warning("no text file input for page %s", page_id)
        segments = set()
        for text_file in text_files:
            basename = os.path.splitext(text_file)[0]
            basename2 = os.path.splitext(basename)[0]
            segment = None
            for id_ in segids:
                if basename.endswith(id_) or basename2.endswith(id_):
                    segment = segids[id_]
                    break
            if not segment:
                self.logger.error("no segment for text file '%s' on page '%s'", text_file, page_id)
                continue
            with open(text_file, 'r') as text_fd:
                text = text_fd.read().strip()
            self.logger.debug("adding '%s' to '%s'", text, segment.id)
            segment.insert_TextEquiv_at(0, TextEquivType(Unicode=text))
            segments.add(segment)
        if not segments.isdisjoint(glyphs):
            nonglyphs = segments.difference(glyphs)
            self.logger.info("updated %d of %d glyphs", len(segments) - len(nonglyphs), len(glyphs))
            segments.difference_update(glyphs)
        if not segments.isdisjoint(words):
            nonwords = segments.difference(words)
            self.logger.info("updated %d of %d words", len(segments) - len(nonwords), len(words))
            segments.difference_update(words)
        if not segments.isdisjoint(lines):
            nonlines = segments.difference(lines)
            self.logger.info("updated %d of %d lines", len(segments) - len(nonlines), len(lines))
            segments.difference_update(lines)
        if not segments.isdisjoint(regions):
            nonregions = segments.difference(regions)
            self.logger.info("updated %d of %d regions", len(segments) - len(nonregions), len(regions))
            segments.difference_update(regions)
        assert len(segments) == 0
        return OcrdPageResult(pcgts)

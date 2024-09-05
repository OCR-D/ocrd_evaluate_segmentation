from __future__ import absolute_import

from typing import Optional
import os
import json
import xlsxwriter

from ocrd_utils import (
    config,
    make_file_id,
    coordinates_of_segment,
    polygon_from_points,
    MIME_TO_EXT
)
from ocrd_modelfactory import page_from_file
from ocrd_models.constants import NAMESPACES
from ocrd_models.ocrd_file import OcrdFileType
from ocrd import Processor


class ExtractLines(Processor):

    @property
    def executable(self):
        return 'ocrd-segment-extract-lines'

    def process_page_file(self, *input_files : Optional[OcrdFileType]) -> None:
        """Extract textline images and texts from the workspace.

        Open and deserialize PAGE input file and their respective images,
        then iterate over the element hierarchy down to the textline level.

        For each textline, if the textline's image size and text length
        do not satisfy ``min-line-height``, ``min-line-width`` or ``min-line-length``,
        then skip.

        Otherwise extract an image (which depending on the workflow can
        already be deskewed, dewarped, binarized etc.), cropped to its
        minimal bounding box, and masked by the coordinate polygon outline.
        Apply ``feature_filter`` (a comma-separated list of image features,
        cf. :py:func:`ocrd.workspace.Workspace.image_from_page`) to skip
        specific features when retrieving derived images.
        If ``transparency`` is true, then also add an alpha channel which is
        fully transparent outside of the mask.

        \b
        If ``output-types`` contains `json`, then create a JSON file with:
        * the IDs of the textline and its parents,
        * the textline's text content,
        * the textline's coordinates relative to the line image,
        * the textline's absolute coordinates,
        * the textline's TextStyle (if any),
        * the textline's @production (if any),
        * the textline's @readingDirection (if any),
        * the textline's @primaryScript (if any),
        * the textline's @primaryLanguage (if any),
        * the textline's AlternativeImage/@comments (features),
        * the parent textregion's @type,
        * the page's @type,
        * the page's DPI value.

        If ``output-types`` contains `text`, then create a plain text file for the content, too.

        If ``output-types`` contains `xlsx`, then (on the page level) create a spreadsheet
        which contains all extracted lines and images in different columns for manual editing.

        \b
        Write all output files in the directory of the output file group, named like so:
        * fileID + regionID + lineID + '.raw.png': line image (if the workflow provides raw images)
        * fileID + regionID + lineID + '.bin.png': line image (if the workflow provides binarized images)
        * fileID + regionID + lineID + '.nrm.png': line image (if the workflow provides grayscale-normalized images)
        * fileID + regionID + lineID + '.json': line metadata.
        * fileID + regionID + lineID + '.gt.txt': line text.
        * fileID + '.xlsx': spreadsheet file.

        (This is intended for correction, training and evaluation of OCR models.)
        """
        library_convention = self.parameter['library-convention']
        textequiv_index = self.parameter['textequiv-index']
        min_line_length = self.parameter['min-line-length']
        min_line_width = self.parameter['min-line-width']
        min_line_height = self.parameter['min-line-height']
        out_types = self.parameter['output-types']

        input_file = input_files[0]
        page_id = input_file.pageId
        try:
            pcgts = page_from_file(input_file)
        except ValueError as err:
            # not PAGE and not an image to generate PAGE for
            self.logger.error(f"non-PAGE input for page {page_id}: {err}")
            raise

        page = pcgts.get_Page()
        page_image, page_coords, page_image_info = self.workspace.image_from_page(
            page, page_id,
            feature_filter=self.parameter['feature_filter'],
            transparency=self.parameter['transparency'])
        if page_image_info.resolution != 1:
            dpi = page_image_info.resolution
            if page_image_info.resolutionUnit == 'cm':
                dpi = round(dpi * 2.54)
        else:
            dpi = None
        ptype = page.get_type()
        file_id = make_file_id(input_file, self.output_file_grp)

        if 'xlsx' in out_types:
            self.logger.info('Writing Excel result file "%s.xlsx" in "%s"', file_id, self.output_file_grp)
            excel_path = '%s.xlsx' % os.path.join(self.output_file_grp, file_id)
            self.workspace.add_file(
                ID=file_id,
                mimetype='application/vnd.ms-excel',
                pageId=page_id,
                local_filename=excel_path,
                file_grp=self.output_file_grp,
                force=config.OCRD_EXISTING_OUTPUT == 'OVERWRITE',
            )
            workbook = xlsxwriter.Workbook(excel_path,
                                           {'strings_to_numbers':  False,
                                            'strings_to_formulas': False,
                                            'strings_to_urls':     False})
            worksheet = workbook.add_worksheet()
            bold = workbook.add_format({'bold': True})
            normal = workbook.add_format({'valign': 'top'})
            editable = workbook.add_format({'valign': 'top'})
            editable.set_locked(False)
            worksheet.set_default_row(height=40)
            worksheet.freeze_panes(1, 0)
            worksheet.write('A1', 'ID', bold)
            worksheet.write('B1', 'Text', bold)
            worksheet.write('C1', 'Status', bold)
            worksheet.write('D1', 'Image', bold)
            symbols = 'ſ ꝛ aͤ oͤ uͤ æ œ Æ Œ ℳ  ç ę ë č ř š ž ě — – - ⸗ = Α α Β β ϐ Γ γ Δ δ Ε ε ϵ Ζ ζ Η η Θ θ ϑ Ι ι ' \
                'Κ κ ϰ Λ λ Μ μ Ν ν Ξ ξ Ο ο Π π ϖ Ρ ρ ϱ Σ σ ς ϲ Τ τ Υ υ ϒ Φ φ ϕ Χ χ Ψ ψ Ω ω'.split(' ')
            for i, s in enumerate(symbols):
                col_idx = 4 + i
                worksheet.write_string(0, col_idx, s, editable)
                worksheet.set_column(col_idx, col_idx, 2)
            worksheet.protect('', {
                'objects':               True,
                'scenarios':             True,
                'format_cells':          False,
                'format_columns':        False,
                'format_rows':           False,
                'insert_columns':        False,
                'insert_rows':           False,
                'insert_hyperlinks':     False,
                'delete_columns':        False,
                'delete_rows':           False,
                'select_locked_cells':   True,
                'sort':                  True,
                'autofilter':            True,
                'pivot_tables':          True,
                'select_unlocked_cells': True,
            })
            url = self._get_presentation_image(input_file, library_convention)
        i = 2
        max_text_length = 0
        regions = page.get_AllRegions(classes=['Text'], order='reading-order')
        if not regions:
            self.logger.warning("Page '%s' contains no text regions", page_id)
        for region in regions:
            region_image, region_coords = self.workspace.image_from_segment(
                region, page_image, page_coords,
                feature_filter=self.parameter['feature_filter'],
                transparency=self.parameter['transparency'])
            rtype = region.get_type()
            lines = region.get_TextLine()
            if not lines:
                self.logger.warning("Region '%s' contains no text lines", region.id)
            for line in lines:
                line_image, line_coords = self.workspace.image_from_segment(
                    line, region_image, region_coords,
                    feature_filter=self.parameter['feature_filter'],
                    transparency=self.parameter['transparency'])
                if not line_image.height or not line_image.width:
                    self.logger.error("Page '%s' line '%s' is of zero height/width", page_id, line.id)
                    continue
                ltext = line.get_TextEquiv()
                if not ltext:
                    self.logger.warning("Line '%s' contains no text content", line.id)
                    ltext = ''
                elif textequiv_index in ['first', 'last']:
                    ltext = ltext[{'first': 0, 'last': -1}[textequiv_index]].Unicode
                else:
                    for textequiv in ltext:
                        if textequiv.index == int(textequiv_index):
                            ltext = textequiv.Unicode
                            break
                    if not isinstance(ltext, str):
                        self.logger.error("Page '%s' line '%s' has no TextEquiv/@index='%s'", page_id, line.id, textequiv_index)
                        continue
                if (len(ltext) < min_line_length or
                    line_image.width < min_line_width or
                    line_image.height < min_line_height):
                    continue
                lpolygon_rel = coordinates_of_segment(
                    line, line_image, line_coords).tolist()
                lpolygon_abs = polygon_from_points(line.get_Coords().points)
                lstyle = line.get_TextStyle() or region.get_TextStyle()
                if lstyle:
                    lstyle = {
                        'fontFamily': lstyle.fontFamily,
                        'fontSize': lstyle.fontSize,
                        'xHeight': lstyle.xHeight,
                        'kerning': lstyle.kerning,
                        'serif': lstyle.serif,
                        'monospace': lstyle.monospace,
                        'bold': lstyle.bold,
                        'italic': lstyle.italic,
                        'smallCaps': lstyle.smallCaps,
                        'letterSpaced': lstyle.letterSpaced,
                        'strikethrough': lstyle.strikethrough,
                        'underlined': lstyle.underlined,
                        'underlineStyle': lstyle.underlineStyle,
                        'subscript': lstyle.subscript,
                        'superscript': lstyle.superscript
                    }
                lfeatures = line_coords['features']
                description = { 'line.ID': line.id,
                                'text': ltext,
                                'style': lstyle,
                                'production': (
                                    line.get_production() or
                                    region.get_production()),
                                'readingDirection': (
                                    line.get_readingDirection() or
                                    region.get_readingDirection() or
                                    page.get_readingDirection()),
                                'primaryScript': (
                                    line.get_primaryScript() or
                                    region.get_primaryScript() or
                                    page.get_primaryScript()),
                                'primaryLanguage': (
                                    line.get_primaryLanguage() or
                                    region.get_primaryLanguage() or
                                    page.get_primaryLanguage()),
                                'features': lfeatures,
                                'DPI': dpi,
                                'coords_rel': lpolygon_rel,
                                'coords_abs': lpolygon_abs,
                                'region.ID': region.id,
                                'region.type': rtype,
                                'page.ID': page_id,
                                'page.type': ptype,
                                'file_grp': self.input_file_grp,
                                'METS.UID': self.workspace.mets.unique_identifier
                }
                if 'binarized' in lfeatures:
                    extension = '.bin'
                elif 'grayscale_normalized' in lfeatures:
                    extension = '.nrm'
                else:
                    extension = '.raw'
                file_path = self.workspace.save_image_file(
                    line_image,
                    file_id + '_' + region.id + '_' + line.id + extension,
                    self.output_file_grp,
                    page_id=page_id,
                    mimetype=self.parameter['mimetype'],
                    force=config.OCRD_EXISTING_OUTPUT == 'OVERWRITE',
                )
                if 'xlsx' in out_types:
                    scale = 40.0 / line_image.height
                    worksheet.write('A%d' % i, file_id + '_' + region.id + '_' + line.id, normal)
                    if len(ltext) > max_text_length:
                        max_text_length = len(ltext)
                        worksheet.set_column('B:B', max_text_length)
                    worksheet.write('B%d' % i, ltext, editable)
                    worksheet.data_validation('C%d' % i, {
                        'validate': 'list', 'source': ['ToDo', 'Done', 'Error']})
                    worksheet.write('C%d' % i, 'ToDo', editable)
                    worksheet.insert_image('D%d' % i, file_path, {
                        'object_position': 1, 'url': url, 'y_scale': scale, 'x_scale': scale})
                file_path = file_path.replace(extension + MIME_TO_EXT[self.parameter['mimetype']], '.json')
                if 'json' in out_types:
                    json.dump(description, open(file_path, 'w'), indent=2)
                file_path = file_path.replace('.json', '.gt.txt')
                if 'text' in out_types:
                    with open(file_path, 'wb') as f:
                        f.write((ltext + '\n').encode('utf-8'))
                i += 1
        if 'xlsx' in out_types:
            workbook.close()

    def _get_presentation_image(self, input_file, library_convention):
        if library_convention == 'slub':
            return self._get_presentation_image_slub(input_file)
        elif library_convention == 'sbb':
            return self._get_presentation_image_sbb(input_file)
        return ''

    def _get_presentation_image_sbb(self, input_file):
        ppn = self.workspace.mets._tree.getroot().xpath('//mods:recordIdentifier[@source="gbv-ppn"]', namespaces=NAMESPACES)[0].text
        pageId = input_file.pageId
        return f'https://digital.staatsbibliothek-berlin.de/werkansicht?PPN={ppn}&PHYSID={pageId}'

    def _get_presentation_image_slub(self, input_file):
        # get Kitodo.Presentation image URL
        url = self.workspace.mets._tree.getroot().xpath(
            '//mets:structMap[@TYPE="LOGICAL"]/mets:div/mets:mptr/@xlink:href',
            namespaces=NAMESPACES)
        NAMESPACES.update({'slub': 'http://slub-dresden.de/'})
        slub = self.workspace.mets._tree.getroot().xpath(
            '//mods:mods/mods:extension/slub:slub',
            namespaces=NAMESPACES)
        if input_file.pageId.startswith('PHYS_'):
            base = '_tif/jpegs/0000' + input_file.pageId[5:] + '.tif.original.jpg'
            if url and url[0].endswith('_anchor.xml'):
                url = url[0]
                url = url[:url.rindex('_anchor.xml')]
                url += base
            elif slub:
                digital = slub[0].xpath('slub:id[@type="digital"]', namespaces=NAMESPACES)
                ats = slub[0].xpath('slub:id[@type="tsl-ats"]', namespaces=NAMESPACES)
                if digital and ats:
                    url = ats[0].text + '_' + digital[0].text
                    url = 'https://digital.slub-dresden.de/data/kitodo/' + url + '/' + url + base
                else:
                    url = ''
        else:
            url = ''
        return url

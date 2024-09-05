from __future__ import absolute_import

import sys
import os
from typing import Optional
import json
from itertools import chain
import click
import numpy as np
from skimage import draw
from PIL import Image
from shapely.geometry import Polygon

from ocrd import Workspace, Processor
from ocrd_utils import (
    getLogger,
    initLogging,
    xywh_from_polygon,
    polygon_from_points,
    coordinates_of_segment,
    MIMETYPE_PAGE
)
from ocrd_models.ocrd_page import parse as parse_page

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.mask import (
    encode as encodeMask,
    merge as mergeMasks,
    area as maskArea
)

class EvaluateSegmentation(Processor):

    @property
    def executable(self):
        return 'ocrd-segment-evaluate'

    def process_workspace(self, workspace: Workspace) -> None:
        """Performs segmentation evaluation with pycocotools on the workspace.

        Open and deserialize PAGE files from the first and second input file group
        (the first as ground truth, the second as prediction).
        Then iterate over the element hierarchy down to ``level-of-operation``.
        Aggregate and convert all pages' segmentation (coordinates and classes)
        to COCO:

        \b
        - On the region level, unless ``ignore-subtype``, differentiate segment
          classes by their `@type`, if applicable.
        - On the region level, unless ``for-categories`` is empty, select only
          segment classes in that (comma-separated) list.
        - If ``only-fg``, then use the foreground mask from the binarized
          image inside each segment for overlap calculations.

        Next, configure and run COCOEval for comparison of all pages. Show the matching
        pairs (GT segment ID, prediction segment ID, IoU) for every overlap on each page.
        Also, calculate per-class precision and recall (at the point of maximum recall).
        Finally, get the typical summary mean average precision / recall (but without
        restriction on the number of segments).

        Write a JSON report to the output file group.
        """
        # region or line level?
        level = self.parameter['level-of-operation']
        onlyfg = self.parameter['only-fg']
        typed = not self.parameter['ignore-subtype']
        selected = self.parameter['for-categories']
        if selected:
            selected = selected.split(',')
        self.workspace = workspace
        self.verify()
        # FIXME: add configurable error handling as in super().process_workspace()
        # get input file groups
        ifgs = self.input_file_grp.split(",")
        # get input file tuples
        ifts = self.zip_input_files(mimetype=MIMETYPE_PAGE, require_first=False)
        # convert to 2 COCO datasets from all page pairs
        categories = ["bg"] # needed by cocoeval
        images = []
        annotations_gt = []
        annotations_dt = []
        for ift in ifts:
            file_gt, file_dt = ift
            if not file_gt:
                self.logger.warning("skipping page %s missing from GT", file_gt.pageId)
                continue
            if not file_dt:
                self.logger.warning("skipping page %s missing from prediction", file_gt.pageId)
                continue
            self.logger.info("processing page %s", file_gt.pageId)
            if self.download:
                file_gt = self.workspace.download_file(file_gt)
                file_dt = self.workspace.download_file(file_dt)
            with pushd_popd(self.workspace.directory):
                pcgts_gt = page_from_file(file_gt)
                pcgts_dt = page_from_file(file_dt)
            page_gt = pcgts_gt.get_Page()
            page_dt = pcgts_dt.get_Page()
            if onlyfg:
                page_image, page_coords, _ = self.workspace.image_from_page(
                    page_gt, file_gt.pageId,
                    feature_selector='binarized',
                    feature_filter='clipped')
                page_mask = ~ np.array(page_image.convert('L'))
            imgid = len(images)
            images.append({'file_name': file_gt.pageId,
                           'width': page_gt.get_imageWidth(),
                           'height': page_gt.get_imageHeight(),
                           })
            # read annotations from each page recursively (all categories including subtypes)
            # and merge GT and prediction categories
            _add_annotations(annotations_gt, page_gt, imgid, categories,
                             level=level, typed=typed,
                             coords=page_coords if onlyfg else None,
                             mask=page_mask if onlyfg else None,
                             log=self.logger)
            _add_annotations(annotations_dt, page_dt, imgid, categories,
                             level=level, typed=typed,
                             coords=page_coords if onlyfg else None,
                             mask=page_mask if onlyfg else None,
                             log=self.logger)

        if level == 'line':
            categories.append('textline')
        elif selected:
            selected = [categories.index(cat) for cat in selected if cat in categories]
        _add_ids(categories)
        _add_ids(images)
        _add_ids(annotations_gt, 1) # cocoeval expects annotation IDs starting at 1
        _add_ids(annotations_dt, 1) # cocoeval expects annotation IDs starting at 1

        self.logger.info(f"found {len(annotations_gt)} GT / {len(annotations_dt)} DT segments"
                         f" in {len(categories) - 1} categories for {len(images)} images")

        coco_gt = _create_coco(categories, images, annotations_gt)
        coco_dt = _create_coco(categories, images, annotations_dt)

        stats = evaluate_coco(coco_gt, coco_dt, self.parameter, selected, log=self.logger)

        # write regions to custom JSON for this page
        file_id = 'id' + self.output_file_grp + '_report'
        workspace.add_file(
            ID=file_id,
            file_grp=self.output_file_grp,
            pageId=None,
            local_filename=os.path.join(self.output_file_grp, file_id + '.json'),
            mimetype='application/json',
            content=json.dumps(stats, indent=2))
        # todo: also write report for each page

@click.command()
@click.option('-G', '--gt-page-filelst', type=click.File('r'),
              help="list file of ground-truth page file paths")
@click.option('-D', '--dt-page-filelst', type=click.File('r'),
              help="list file of detection page file paths")
@click.option('-I', '--bin-img-filelst', type=click.File('r'),
              help="list file of binarized image file paths")
@click.option('-L', '--level-of-operation', type=click.Choice(['region', 'line']), default='region',
              help="hierarchy level of segments to compare")
@click.option('-T', '--ignore-subtype', is_flag=True,
              help="on region level, ignore @type distinction")
@click.option('-C', '--for-categories', default='', type=str,
              help="on region level, comma-separated list of category names to evaluate (empty for all)")
@click.option('-R', '--report-file', type=click.File('w'), default="eval.log",
              help="file name to write evaluation results to")
@click.argument('tabfile', type=click.File('r'), required=False)
def standalone_cli(gt_page_filelst,
                   dt_page_filelst,
                   bin_img_filelst,
                   level_of_operation,
                   ignore_subtype,
                   for_categories,
                   report_file,
                   tabfile):
    """Performs segmentation evaluation with pycocotools on the given PAGE-XML files.

    \b
    Open and deserialize PAGE files from the list files.
    Then iterate over the element hierarchy down to ``level-of-operation``.
    Aggregate and convert all pages' segmentation (coordinates and classes)
    to COCO:

    \b
    - On the region level, unless ``ignore-subtype``, differentiate segment
      classes by their `@type`, if applicable.
    - On the region level, unless ``for-categories`` is empty, select only
      segment classes in that (comma-separated) list.
    - If image files are given (as separate file list or in the 3rd column
      of the tab-separated list file), then for each PAGE file pair, use
      the foreground mask from the binarized image inside all segments for
      overlap calculations.

    \b
    Next, configure and run COCOEval for comparison of all pages. Show the 
    matching pairs (GT segment ID, prediction segment ID, IoU) for every
    overlap on each page.
    Also, calculate per-class precision and recall (at maximum recall).
    Finally, get the typical summary mean average precision / recall
    (but without restriction on the number of segments), and write all
    statistics to ``report-file``.

    \b
    Write a JSON report to the output file group.
    """
    initLogging()
    assert (tabfile is None) == (gt_page_filelst is not None) == (dt_page_filelst is not None), \
        "pass file lists either as tab-separated single file or as separate files"
    if tabfile is None:
        gt_page_files = [line.strip() for line in gt_page_filelst.readlines()]
        dt_page_files = [line.strip() for line in dt_page_filelst.readlines()]
        assert len(gt_page_files) == len(dt_page_files), \
            "number of DT files must match number of GT files"
        if bin_img_filelst is not None:
            bin_img_files = [line.strip() for line in bin_img_filelst.readlines()]
            assert len(bin_img_files) == len(gt_page_files), \
                "number of image files must match number of GT files"
        else:
            bin_img_files = None
    else:
        files = [line.strip().split('\t') for line in tabfile.readlines()]
        assert len(files), "list of files is empty"
        len0 = len(files[0])
        assert 2 <= len0 <= 3, "list of files must be tab-separated (GT, DT[, bin-img])"
        assert all(map(lambda line: len(line) == len0, files)), \
            "number of DT files must match number of GT files"
        if len0 == 2:
            gt_page_files, dt_page_files = zip(*files)
            bin_img_files = None
        else:
            gt_page_files, dt_page_files, bin_img_files = zip(*files)
    stats = evaluate_files(gt_page_files,
                           dt_page_files,
                           bin_img_files,
                           level_of_operation,
                           not ignore_subtype,
                           for_categories)
    json.dump(stats, report_file, indent=2)

# standalone entry point
def evaluate_files(gt_files, dt_files, img_files=None, level='region', typed=True, selected=None):
    log = getLogger('EvaluateSegmentation')
    categories = ["bg"] # needed by cocoeval
    images = []
    annotations_gt = []
    annotations_dt = []
    for gt_file, dt_file, img_file in zip(gt_files, dt_files,
                                          img_files or [None] * len(gt_files)):
        pcgts_gt = parse_page(gt_file)
        pcgts_dt = parse_page(dt_file)
        page_id = pcgts_gt.pcGtsId or gt_file
        log.info("processing page %s", page_id)
        page_gt = pcgts_gt.get_Page()
        page_dt = pcgts_dt.get_Page()
        if img_file:
            page_image = Image.open(img_file)
            assert page_image.mode == '1', "input images must already be binarized"
            assert page_image.width - 2 < page_gt.get_imageWidth() < page_image.width + 2, \
                "mismatch between width of binary image and PAGE description"
            assert page_image.height - 2 < page_gt.get_imageHeight() < page_image.height + 2, \
                "mismatch between height of binary image and PAGE description"
            page_mask = ~ np.array(page_image)
            page_coords = {"transform": np.eye(3), "angle": 0, "features": "binarized"}
        imgid = len(images)
        images.append({'file_name': page_id,
                       'width': page_gt.get_imageWidth(),
                       'height': page_gt.get_imageHeight(),
        })
        # read annotations from each page recursively (all categories including subtypes)
        # and merge GT and prediction categories
        _add_annotations(annotations_gt, page_gt, imgid, categories,
                         level=level, typed=typed,
                         coords=page_coords if img_file else None,
                         mask=page_mask if img_file else None,
                         log=log)
        _add_annotations(annotations_dt, page_dt, imgid, categories,
                         level=level, typed=typed,
                         coords=page_coords if img_file else None,
                         mask=page_mask if img_file else None,
                         log=log)

    if level == 'line':
        categories.append('textline')
    elif selected:
        selected = [categories.index(cat) for cat in selected if cat in categories]
    _add_ids(categories)
    _add_ids(images)
    _add_ids(annotations_gt, 1) # cocoeval expects annotation IDs starting at 1
    _add_ids(annotations_dt, 1) # cocoeval expects annotation IDs starting at 1

    log.info(f"found {len(annotations_gt)} GT / {len(annotations_dt)} DT segments"
             f" in {len(categories) - 1} categories for {len(images)} images")

    coco_gt = _create_coco(categories, images, annotations_gt)
    coco_dt = _create_coco(categories, images, annotations_dt)

    parameters = {"level-of-operation": level,
                  "only-fg": bool(img_files),
                  "ignore-subtype": not typed,
                  "for-categories": selected}
    stats = evaluate_coco(coco_gt, coco_dt, parameters, selected)
    return stats

def evaluate_coco(coco_gt, coco_dt, parameters, catIds=None, log=None):
    if log is None:
        log = getLogger('EvaluateSegmentation')
    log.info("comparing segmentations")
    stats = dict(parameters)
    coco_eval = COCOeval(coco_gt, coco_dt, 'segm') # bbox
    if catIds:
       coco_eval.params.catIds = catIds
    #coco_eval.params.iouThrs = [.5:.05:.95]
    #coco_eval.params.iouThrs = np.linspace(.3, .95, 14)
    coco_eval.params.maxDets = [None] # unlimited nr of detections (requires pycocotools#559)
    #coco_eval.params.areaRng = [(0, np.inf)] # unlimited region size
    #coco_eval.params.areaRngLbl = ['all'] # unlimited region size
    # Note: The IoU threshold criterion is inadequate for flat segmentation,
    #       because over-/undersegmentation can quickly become false negative/positive.
    #       The pycocotools implementation is especially inadequate, because
    #       it only counts 1:1 matches (and not even the largest or best-scoring, #564).
    #       On the other hand, purely pixel-wise measures do not distinguish instances,
    #       i.e. neighbours can quickly become merged or instances torn apart.
    #       Our approach therefore does not build on pycocotools for matching
    #       and aggregation, only for fast IoU calculation. All non-zero pairs
    #       are considered matches if their intersection over union > 0.5 _or_
    #       their intersection over either side > 0.5. Matches can thus be n:m.
    #       Non-matches are counted as well (false positives and false negatives).
    #       Aggregation uses microaveraging over images. Besides counting segments,
    #       the pixel areas are counted and averaged (as ratios).
    # FIXME: We must differentiate between allowable and non-allowable over/under-segmentation (splits/merges).
    #        (A region's split is allowable if it flows in the textLineOrder of the respective GT,
    #         i.e. lines are likely to be either on one side or the other, but not both.
    #         For top-to-bottom/bottom-to-top regions, vertical splits are allowable.
    #         For left-to-right/right-to-left regions, horizontal splits are allowable.
    #         To be sure, we could also validate that explicitly – evaluating both levels at the same time.
    #         Analogously, a number of regions' merge is allowable if it flows in the textLineOrder
    #         of them all, and the GT global reading order has no other regions in between.
    #         For top-to-bottom/bottom-to-top regions, vertical merges are allowable.
    #         For left-to-right/right-to-left regions, horizontal merges are allowable.
    #         Again, we could also validate that the overall textline flow is equivalent.)
    #        This difference can in turn be used to weigh a match pair's score accordingly
    #        when aggregating. For precision-like scores, we would rule out non-allowable merges
    #        (by counting them as FP), and for recall-like scores, we would rule out non-allowable splits
    #        (by counting them as FN).
    #        We can also weigh these non-allowable cases by their share of height
    #        (in vertical textLineOrder and horizontal writing) or width
    #        (in horizontal textLineOrder and vertical writing) which is in disagreement,
    #        or the share of its textlines that have been split or lost.
    #        Furthermore, we can weigh matches by the share of non-text regions or fg pixels involved.
    coco_eval.evaluate()
    # get by-page alignment (ignoring inadequate 1:1 matching by pycocotools)
    def get(arg):
        return lambda x: x[arg]
    numImgs = len(coco_eval.params.imgIds)
    numAreas = len(coco_eval.params.areaRng)
    for imgind, imgId in enumerate(coco_eval.params.imgIds):
        img = coco_gt.imgs[imgId]
        pageId = img['file_name']
        for catind, catId in enumerate(coco_eval.params.catIds):
            cat = coco_gt.cats[catId]
            catName = cat['name']
            if not catId:
                continue
            # bypassing COCOeval.evaluateImg, hook onto its results
            # (again, we stay at areaRng[0]=all and maxDets[0]=all)
            start = catind * numImgs * numAreas
            evalimg = coco_eval.evalImgs[start + imgind]
            if evalimg is None:
                continue # no DT and GT here
            # record as dict by pageId / by category
            imgstats = stats.setdefault('by-image', dict())
            pagestats = imgstats.setdefault(pageId, dict())
            # get matches and ious and scores
            ious = coco_eval.ious[imgId, catId]
            if len(ious):
                overlaps_dt, overlaps_gt = ious.nonzero()
            else:
                overlaps_dt = overlaps_gt = []
            # reconstruct score sorting in computeIoU
            gt = coco_eval._gts[imgId, catId]
            dt = coco_eval._dts[imgId, catId]
            dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
            dt = [dt[i] for i in dtind]
            matches = list()
            gtmatches = dict()
            dtmatches = dict()
            for dtind, gtind in zip(overlaps_dt, overlaps_gt):
                d = dt[dtind]
                g = gt[gtind]
                iou = ious[dtind, gtind]
                union = maskArea(mergeMasks([g['segmentation'], d['segmentation']]))
                intersection = int(iou * union)
                # cannot use g or d['area'] here, because mask might be fractional (only-fg) instead of outline
                areag = int(maskArea(g['segmentation']))
                aread = int(maskArea(d['segmentation']))
                iogt = intersection / areag
                iodt = intersection / aread
                if iou < 0.5 and iogt < 0.5 and iodt < 0.5:
                    continue
                gtmatches.setdefault(gtind, list()).append(dtind)
                dtmatches.setdefault(dtind, list()).append(gtind)
                matches.append((g['id'],
                                d['id'],
                                iogt, iodt, iou, intersection))
                pagestats.setdefault('true_positives', dict()).setdefault(catName, list()).append(
                    {'GT.ID': g['segment_id'],
                     'DT.ID': d['segment_id'],
                     'GT.area': areag,
                     'DT.area': aread,
                     'I.area': intersection,
                     'IoGT': iogt,
                     'IoDT': iodt,
                     'IoU': iou})
            dtmisses = []
            for dtind, d in enumerate(dt):
                if dtind in dtmatches:
                    continue
                dtmisses.append((d['id'], maskArea(d['segmentation'])))
                pagestats.setdefault('false_positives', dict()).setdefault(catName, list()).append(
                    {'DT.ID': d['segment_id'],
                     'area': int(d['area'])})
            gtmisses = []
            for gtind, g in enumerate(gt):
                if gtind in gtmatches:
                    continue
                gtmisses.append((g['id'], maskArea(g['segmentation'])))
                pagestats.setdefault('false_negatives', dict()).setdefault(catName, list()).append(
                    {'GT.ID': g['segment_id'],
                     'area': int(g['area'])})
            # measure under/oversegmentation for this image and category
            # (follows Zhang et al 2021: Rethinking Semantic Segmentation Evaluation [arXiv:2101.08418])
            over_gt = set(gtind for gtind in gtmatches if len(gtmatches[gtind]) > 1)
            over_dt = set(chain.from_iterable(
                gtmatches[gtind] for gtind in gtmatches if len(gtmatches[gtind]) > 1))
            under_dt = set(dtind for dtind in dtmatches if len(dtmatches[dtind]) > 1)
            under_gt = set(chain.from_iterable(
                dtmatches[dtind] for dtind in dtmatches if len(dtmatches[dtind]) > 1))
            over_degree = sum(len(gtmatches[gtind]) - 1 for gtind in gtmatches)
            under_degree = sum(len(dtmatches[dtind]) - 1 for dtind in dtmatches)
            if len(dt) and len(gt):
                oversegmentation = len(over_gt) * len(over_dt) / len(gt) / len(dt)
                undersegmentation = len(under_gt) * len(under_dt) / len(gt) / len(dt)
                # Zhang's idea of attenuating the under/oversegmentation ratio with a "penalty"
                # to account for the degree of further sub-segmentation is misguided IMHO,
                # because its degree term depends on the total number of segments:
                # oversegmentation = np.tanh(oversegmentation * over_degree)
                # undersegmentation = np.tanh(undersegmentation * under_degree)
                pagestats.setdefault('oversegmentation', dict())[catName] = oversegmentation
                pagestats.setdefault('undersegmentation', dict())[catName] = undersegmentation
                pagestats.setdefault('precision', dict())[catName] =  (len(dt) - len(dtmisses)) / len(dt)
                pagestats.setdefault('recall', dict())[catName] =  (len(gt) - len(gtmisses)) / len(gt)
            tparea = sum(map(get(5), matches)) # sum(inter)
            fparea = sum(map(get(1), dtmisses)) # sum(area)
            fnarea = sum(map(get(1), gtmisses)) # sum(area)
            if tparea or (fparea and fnarea):
                pagestats.setdefault('pixel_precision', dict())[catName] = tparea / (tparea + fparea)
                pagestats.setdefault('pixel_recall', dict())[catName] =  tparea / (tparea + fnarea)
                pagestats.setdefault('pixel_iou', dict())[catName] =  tparea / (tparea + fparea + fnarea)
            # aggregate per-img/per-cat IoUs for microaveraging
            evalimg['matches'] = matches # TP
            evalimg['dtMisses'] = dtmisses # FP
            evalimg['gtMisses'] = gtmisses # FN
            evalimg['dtIdsOver'] = [dt[dtind]['id'] for dtind in over_dt]
            evalimg['gtIdsOver'] = [gt[gtind]['id'] for gtind in over_gt]
            evalimg['dtIdsUnder'] = [dt[dtind]['id'] for dtind in under_dt]
            evalimg['gtIdsUnder'] = [gt[gtind]['id'] for gtind in under_gt]

    catstats = stats.setdefault('by-category', dict())
    # accumulate our over-/undersegmentation and IoU ratios
    numImgs = len(coco_eval.params.imgIds)
    numAreas = len(coco_eval.params.areaRng)
    for catind, catId in enumerate(coco_eval.params.catIds):
        cat = coco_gt.cats[catId]
        catstats.setdefault(cat['name'], dict())
        start = catind * numImgs * numAreas
        # again, we stay at areaRng[0]=all and maxDets[0]=all
        evalimgs = [coco_eval.evalImgs[start + imgind] for imgind in range(numImgs)]
        evalimgs = [img for img in evalimgs if img is not None]
        assert all(img['category_id'] == catId for img in evalimgs)
        assert all(img['maxDet'] is None for img in evalimgs)
        assert all(img['aRng'] == coco_eval.params.areaRng[0] for img in evalimgs)
        if not len(evalimgs):
            continue
        # again, we can ignore gtIgnore here, because we only look at areaRng[0]=all
        # again, we can ignore dtIgnore here, because we only look at maxDet=None
        numDTs = sum(len(img['dtIds']) for img in evalimgs)
        numGTs = sum(len(img['gtIds']) for img in evalimgs)
        overDTs = sum(len(img['dtIdsOver']) for img in evalimgs)
        overGTs = sum(len(img['gtIdsOver']) for img in evalimgs)
        underDTs = sum(len(img['dtIdsUnder']) for img in evalimgs)
        underGTs = sum(len(img['gtIdsUnder']) for img in evalimgs)
        numIoUs = sum(len(img['matches']) for img in evalimgs)
        numFPs = sum(len(img['dtMisses']) for img in evalimgs)
        numFNs = sum(len(img['gtMisses']) for img in evalimgs)
        sumIoUs = sum(sum(map(get(4), img['matches'])) for img in evalimgs) # sum(iou)
        sumIoGTs = sum(sum(map(get(2), img['matches'])) for img in evalimgs) # sum(iogt)
        sumIoDTs = sum(sum(map(get(3), img['matches'])) for img in evalimgs) # sum(iodt)
        sumTParea = sum(sum(map(get(5), img['matches'])) for img in evalimgs) # sum(inter)
        sumFParea = sum(sum(map(get(1), img['dtMisses'])) for img in evalimgs) # sum(area)
        sumFNarea = sum(sum(map(get(1), img['gtMisses'])) for img in evalimgs) # sum(area)
        if numDTs and numGTs:
            oversegmentation = overDTs * overGTs / numDTs / numGTs
            undersegmentation = underDTs * underGTs / numDTs / numGTs
            precision = (numDTs - numFPs) / numDTs
            recall = (numGTs - numFNs) / numGTs
        else:
            oversegmentation = undersegmentation = precision = recall = -1
        if numIoUs:
            iou = sumIoUs / numIoUs
            iogt = sumIoGTs / numIoUs
            iodt = sumIoDTs / numIoUs
        else:
            iou = iogt = iodt = -1
        if sumTParea or (sumFParea and sumFNarea):
            pixel_precision = sumTParea / (sumTParea + sumFParea)
            pixel_recall = sumTParea / (sumTParea + sumFNarea)
            pixel_iou = sumTParea / (sumTParea + sumFParea + sumFNarea)
        else:
            pixel_precision = pixel_recall = pixel_iou = -1
        catstats[cat['name']]['oversegmentation'] = oversegmentation
        catstats[cat['name']]['undersegmentation'] = undersegmentation
        catstats[cat['name']]['segment-precision'] = precision
        catstats[cat['name']]['segment-recall'] = recall
        catstats[cat['name']]['IoGT'] = iogt # i.e. per-match pixel-recall
        catstats[cat['name']]['IoDT'] = iodt # i.e. per-match pixel-precision
        catstats[cat['name']]['IoU'] = iou # i.e. per-match pixel-jaccardindex
        catstats[cat['name']]['pixel-precision'] = pixel_precision
        catstats[cat['name']]['pixel-recall'] = pixel_recall
        catstats[cat['name']]['pixel-iou'] = pixel_iou

    coco_eval.accumulate()
    coco_eval.summarize()
    statInds = np.ones(12, bool)
    statInds[7] = False # AR maxDet[1]
    statInds[8] = False # AR maxDet[2]
    coco_eval.stats = coco_eval.stats[statInds]
    stats['scores'] = dict(zip([
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all ]',
        'Average Precision  (AP) @[ IoU=0.50      | area=   all ]',
        'Average Precision  (AP) @[ IoU=0.75      | area=   all ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large ]',
        ], coco_eval.stats.tolist()))
    return stats

def _create_coco(categories, images, annotations):
    coco = COCO()
    coco.dataset = {'categories': categories, 'images': images, 'annotations': annotations}
    with NoStdout():
        coco.createIndex()
    return coco

def _add_annotations(annotations, page, imgid, categories,
                     level='region', typed=True, coords=None, mask=None, log=None):
    for region in page.get_AllRegions(classes=None if level == 'region' else ['Text']):
        if level == 'region':
            cat = region.__class__.__name__[:-4]
            if typed and hasattr(region, 'get_type') and region.get_type():
                cat += '.' + region.get_type()
            if cat not in categories:
                categories.append(cat)
            catid = categories.index(cat)
            _add_annotation(annotations, region, imgid, catid,
                            coords=coords, mask=mask, log=log)
            continue
        for line in region.get_TextLine():
            _add_annotation(annotations, line, imgid, 1,
                            coords=coords, mask=mask, log=log)

def _add_annotation(annotations, segment, imgid, catid, coords=None, mask=None, log=None):
    if log is None:
        log = getLogger('EvaluateSegmentation')
    score = segment.get_Coords().get_conf() or 1.0
    polygon = polygon_from_points(segment.get_Coords().points)
    if len(polygon) < 3:
        log.warning('ignoring segment "%s" with only %d points', segment.id, len(polygon))
        return
    xywh = xywh_from_polygon(polygon)
    if mask is None:
        segmentation = np.array(polygon).reshape(1, -1).tolist()
    else:
        polygon = coordinates_of_segment(segment, None, coords)
        py, px = draw.polygon(polygon[:,1], polygon[:,0], mask.shape)
        masked = np.zeros(mask.shape, dtype=np.uint8, order='F') # pycocotools.mask wants Fortran-contiguous arrays
        masked[py, px] = 1 * mask[py, px]
        segmentation = encodeMask(masked)
    annotations.append(
        {'segment_id': segment.id, # non-standard string-valued in addition to 'id'
         'image_id': imgid,
         'category_id': catid,
         'segmentation': segmentation,
         'area': Polygon(polygon).area,
         'bbox': [xywh['x'], xywh['y'], xywh['w'], xywh['h']],
         'score': score,
         'iscrowd': 0})

def _add_ids(entries, start=0):
    for i, entry in enumerate(entries, start):
        if isinstance(entry, dict):
            entry['id'] = i
        else:
            entries[i] = {'id': i, 'name': entry}

class NoStdout():
    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, etype, evalue, etrace):
        sys.stdout = self.stdout
        if etype is not None:
            return False # reraise

    def write(self, value):
        pass

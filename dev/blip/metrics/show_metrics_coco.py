from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

# coco = COCO("../../vizwiz/annotations/val.json")
coco = COCO("output.json")

cocoRes = coco.loadRes("results_batch.json")

# 4. Считаем метрики
cocoEval = COCOEvalCap(coco, cocoRes)
cocoEval.evaluate()

# 5. Выводим результаты
for metric, score in cocoEval.eval.items():
    print(f"{metric}: {score:.4f}")
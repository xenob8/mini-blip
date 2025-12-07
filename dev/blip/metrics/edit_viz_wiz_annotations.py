import json

with open("../../vizwiz/annotations/val.json") as f:
    new = {}
    fi = json.loads(f.read())
    new["info"] = fi["info"]
    new["images"] = []
    new["annotations"] = []
    images = fi["images"]
    for img in images:
        if img["id"] > 23480:
            continue
        else:
            new["images"].append(img)

    anot = fi["annotations"]
    for an in anot:
        if an["image_id"] > 23480:
            continue
        else:
            new["annotations"].append(an)

    with open("output.json", "w") as json_file:
        # Use json.dump() to write the data to the file
        # 'indent' argument makes the output more readable by adding indentation
        json.dump(new, json_file, indent=4)


    print()
import csv

class Labeler:
    def __init__(self, text_objs, base_task):
        self.text_objs = text_objs
        self.base_task = base_task

    def reset_labels(self):
        """Reset all labels to None."""
        for text_obj in self.text_objs:
            text_obj.label = None

    def auto_label(self):
        for text_obj in self.text_objs:
            if text_obj.category in ["different_tasks", "others"]:
                text_obj.label = 0

    def export_to_csv(self, filename):
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Text", "Category", "Label"])  # Header
            for obj in self.text_objs:
                writer.writerow([obj.text, obj.category, obj.label])

    def import_from_csv(self, filename):
        with open(filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # skip header row
            for row, obj in zip(reader, self.text_objs):
                _, _, label = row
                if label:  # Check if the label is not empty
                    obj.label = int(label)
                else:
                    print(f"Warning: Skipped an empty label for text: {obj.text}")


    def run(self):
        self.auto_label()
        self.export_to_csv('../results/texts_for_labeling.csv')
        # You will manually label using the exported CSV here.
        input("Please manually label the data in 'texts_for_labeling.csv'. Press Enter when done.")
        self.import_from_csv('../results/texts_for_labeling.csv')

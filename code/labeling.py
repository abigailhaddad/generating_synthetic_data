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
            if text_obj.category == "different_tasks" or text_obj.category == "others":
                text_obj.label = 0

    def manual_label(self):
        for obj in self.text_objs:
            # If it's already labeled or it's from "different_tasks" or "others", skip it
            if obj.label is not None or obj.category in ["different_tasks", "others"]:
                continue
            
            # Display the text
            print(obj.text)
            
            # Ask for a label
            label = input("Is this explaining the base task? (1 for Yes, 0 for No, s to skip): ")
            if label == 's':
                continue
            
            obj.label = int(label)

    def run(self):
        self.auto_label()
        self.manual_label()

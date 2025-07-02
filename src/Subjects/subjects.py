import json
from .subject import Subject

class Subjects:
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            self._data = json.load(f)

    def get_subject(self, his_id):
        data = self._data.get(his_id)
        if data is None:
            return None
        return Subject(subject_id=his_id, data=data)
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, index):
        if isinstance(index, int):
            his_id = list(self._data.keys())[index]
            return self.get_subject(his_id)
        elif isinstance(index, str):
            return self.get_subject(index)
        else:
            raise TypeError("Index must be an integer or a string (his_id).")
        
    def __iter__(self):
        for his_id in self._data:
            yield self.get_subject(his_id)

    def __contains__(self, his_id):
        return his_id in self._data
    
    def __repr__(self):
        return f"Subjects({len(self._data)} subjects loaded)"
    
    def __str__(self):
        return f"Subjects with {len(self._data)} subjects: {', '.join(self._data.keys())}"
    
    def to_dict(self):
        return self._data
    
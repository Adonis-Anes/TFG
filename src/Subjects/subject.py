from datetime import date


class Subject:
    def __init__(self, subject_id=None, data=None):
        self.id = None
        self.his_id = subject_id
        self.first_name = None
        self.last_name = None
        self.birthday = None
        self.sex = None
        self.hand = None
        self.ref_channel = None
        if data:
            self.load_from_dict(data)

    def load_from_dict(self, data):
        self.id = data.get('id')
        self.his_id = data.get('his_id', self.his_id)
        self.first_name = data.get('first_name')
        self.last_name = data.get('last_name')
        self.birthday = data.get('birthday')
        self.sex = data.get('sex')
        self.hand = data.get('hand')
        self.ref_channel = data.get('ref_channel')

    def to_mne_dict(self):
        mne_dict = {
            'id': self.id,
            'his_id': self.his_id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'birthday': date(
                year=self.birthday['year'],
                month=self.birthday['month'],
                day=self.birthday['day']),
            'sex': self.sex,
            'hand': self.hand
        }
        return mne_dict

import pandas as pd

DROP = [
    "Weight (kg)",
    "Height (cm)",
    "Society of Thoracic Surgoens STS Score (%)",
    "Peak Pulmonary Artery pressure (mmHg)",
    "Mean Pulmonary Artery pressure (mmHg)",
    "Post-procedural aortic insufficiency",
    "Post-procedural aortic peak gradient (mmHg)",
    "Post-procedural aortic mean gradient (mmHg)",
    "Access Route",
    "Valve Type",
    "Systolic BP (mmHg)",
    "Diastolic BP (mmHg)",
    "Heart Rate (bpm)",
    'PUID',
    "EUID",
    'Access Route',
    'Heart Failure Event Date',
    'Heart failure admission event',
    'Death event',
    'Death date',
    'Pateint last seen date',
    'End Point',
    'folder',
    'Dataset_type',
    'Unnamed: 0',
    'TAVR Surgery Date',
    'Post-procedural aortic insufficiency',
    'BMI (kg/m^2)',
    'eGFR',
    'Creatinine',
    'CCS class',
    'NYHA functional class',
    'Baseline LV mass index (g/m2)',
    'Baseline LA volume index (ml/m2)',
    'Baseline Aortic valve peak gradient (mmHg)',
    'Baseline Aortic valve mean gradient (mmHg)',
    'Baseline Aortic valve area index (cm2/m2)',
    'Baseline TAPSE (mm)',
    'Baseline Degree of Mitral regurgitation',
    'Baseline Degree of tricuspid regurgitation',
    'Global peak endocardial minimum principal strain (%)',
    'Endocardial minimum principal strain Segement 1 (%)',
    'Endocardial minimum principal strain Segement 2 (%)',
    'Endocardial minimum principal strain Segement 3 (%)',
    'Endocardial minimum principal strain Segement 4 (%)',
    'Endocardial minimum principal strain Segement 5 (%)',
    'Endocardial minimum principal strain Segement 6 (%)',
    'Endocardial minimum principal strain Segement 7 (%)',
    'Endocardial minimum principal strain Segement 8 (%)',
    'Endocardial minimum principal strain Segement 9 (%)',
    'Endocardial minimum principal strain Segement 10 (%)',
    'Endocardial minimum principal strain Segement 11 (%)',
    'Endocardial minimum principal strain Segement 12 (%)',
    'Endocardial minimum principal strain Segement 13 (%)',
    'Endocardial minimum principal strain Segement 14 (%)',
    'Endocardial minimum principal strain Segement 15 (%)',
    'Endocardial minimum principal strain Segement 16 (%)',
    'Endocardial minimum principal strain Segement 17 (%)',
    'Global peak epicardial minimum principal strain (%)',
    'Epicardial minimum principal strain Segement 1 (%)',
    'Epicardial minimum principal strain Segement 2 (%)',
    'Epicardial minimum principal strain Segement 3 (%)',
    'Epicardial minimum principal strain Segement 4 (%)',
    'Epicardial minimum principal strain Segement 5 (%)',
    'Epicardial minimum principal strain Segement 6 (%)',
    'Epicardial minimum principal strain Segement 7 (%)',
    'Epicardial minimum principal strain Segement 8 (%)',
    'Epicardial minimum principal strain Segement 9 (%)',
    'Epicardial minimum principal strain Segement 10 (%)',
    'Epicardial minimum principal strain Segement 11 (%)',
    'Epicardial minimum principal strain Segement 12 (%)',
    'Epicardial minimum principal strain Segement 13 (%)',
    'Epicardial minimum principal strain Segement 14 (%)',
    'Epicardial minimum principal strain Segement 15 (%)',
    'Epicardial minimum principal strain Segement 16 (%)',
    'Epicardial minimum principal strain Segement 17 (%)',
    'Global peak transmural minimum principal strain (%)',
    'Transmural minimum principal strain Segment 1 (%)',
    'Transmural minimum principal strain Segment 2 (%)',
    'Transmural minimum principal strain Segment 3 (%)',
    'Transmural minimum principal strain Segment 4 (%)',
    'Transmural minimum principal strain Segment 5 (%)',
    'Transmural minimum principal strain Segment 6 (%)',
    'Transmural minimum principal strain Segment 7 (%)',
    'Transmural minimum principal strain Segment 8 (%)',
    'Transmural minimum principal strain Segment 9 (%)',
    'Transmural minimum principal strain Segment 10 (%)',
    'Transmural minimum principal strain Segment 11 (%)',
    'Transmural minimum principal strain Segment 12 (%)',
    'Transmural minimum principal strain Segment 13 (%)',
    'Transmural minimum principal strain Segment 14 (%)',
    'Transmural minimum principal strain Segment 15 (%)',
    'Transmural minimum principal strain Segment 16 (%)',
    'Transmural minimum principal strain Segment 17 (%)'
]


class MetadataProcessing:
    def __init__(self, metadata_path: str):
        metadata = pd.read_csv(metadata_path)
        metadata.columns = metadata.columns.str.strip()
        metadata = metadata.drop(['EndPointFinalized'], axis=1)
        MetadataProcessing._fill_nan(metadata)
        metadata = metadata.drop(DROP, axis=1)
        self.metadata, self.encoders = MetadataProcessing.encode_categorical(metadata)

    @staticmethod
    def encode_categorical(metadata):
        from sklearn.preprocessing import LabelEncoder

        label_encoders = {}
        for column in metadata.columns:
            # check the datatype of the column, and if it is an object, encode it
            if metadata[column].dtype == object:
                label_encoders[column] = LabelEncoder()
                metadata[column] = label_encoders[column].fit_transform(metadata[column])
        return metadata, label_encoders

    @staticmethod
    def _fill_nan(metadata):
        fill_by_mean = [
            'Mean Pulmonary Artery pressure (mmHg)',
            'Peak Pulmonary Artery pressure (mmHg)',
            'Baseline TAPSE (mm)'
        ]
        fill_by_mode = [
            'Baseline Degree of Mitral regurgitation',
            'Baseline Degree of tricuspid regurgitation'
        ]
        for column in fill_by_mean:
            metadata[column] = metadata[column].fillna(metadata[column].mean())

        for column in fill_by_mode:
            metadata[column] = metadata[column].fillna(metadata[column].mode()[0])

    def get_case_metadata(self, case_name: str):
        patient_id = int(case_name.split("_")[1])
        return self.metadata.loc[self.metadata['id'] == patient_id].drop("id", axis=1).values[0]

    def __repr__(self):
        return str(self.metadata.columns)


if __name__ == "__main__":
    p = MetadataProcessing("/home/andrewheschl/PycharmProjects/Cardiac_TAVR_Classification/mapped_ids_data.csv")
    print(p.get_case_metadata("case_00124"))

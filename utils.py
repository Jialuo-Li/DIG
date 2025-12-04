import os
import json
from torch.utils.data import Dataset

API_URL = "http://localhost:8000/v1"
API_KEY = "token-abc123"
MODEL_NAME = os.environ.get("MODEL_NAME", "")

QUERY_IDENTIFICATION_PROMPT = """You are a helpful assistant in a video-based question-answering process. 

**Core Task & Definitions**

You will classify the given query into one of two categories:

1. Global Query (isGlobal: true): The query requires going through and understanding the entire video content.

2. Localized Query (isGlobal: false): The query that can be fully answered by extracting and analyzing several specific segments within the video.
   
**Instructions for Analysis and Response**

In your analysis, please follow this structured reasoning process to classify the query:

Step 1. Understand the Query: First, read the query to understand its general meaning and core intent.

Step 2. Infer Video Style (Hypothetically): Based on the query's phrasing, make a reasonable inference about the style of the video (e.g., is it a narrative film, an educational lesson, a documentary, etc.)?

Step 3. Identify Referents: Analyze if the query has specific referents. A referent is an entity (person, object), action, event, or even specific piece of information, depending on the type of video you inferred. For instance, in 'What does Professor Smith write about quantum physics?', the referent is 'Professor Smith' and 'quantum physics' since the video style is likely a lesson.

Step 4. Evaluate Referents in Context: Based on the results from step 3 and the criteria below, determine whether the query is Global or Localized.

    (i) The query is Global if it meets either condition: 1. Lacks a specific referent. The examples include: Summary-based: "primary focus," "in summary," "what is the video about,"; Negations: "what was not mentioned" 2. Has a referent, but answering still requires a holistic understanding from going through the entire video. The examples include: "what is the boy's overall role?"

    (ii) The query is Localized if it has specific referents, and the answer can be found by focusing on specific, related segments where it appears. The examples include:
        Entity-based: "the person in the red shirt," "the black dog," "Professor Smith," "the little girl."
        Action/Event-based: "what is [X] doing," "how does [X] build,"
        Temporal/Sequential: "at the beginning," "after the explosion," 

Please provide your answer in the following format: {"analysis_step1": str, "analysis_step2": str, "analysis_step3": str, "analysis_step4": str, "isGlobal": true/false}

User Query: <<<question>>>"""


REWARD_ASSIGNMENT_PROMPT = """You are a reward model for a video-based question-answering system.

**Task**

You will receive a question and a sampled video frame. Your task is to evaluate the relevance of this frame for answering the question. Please assign a reward score that indicates how useful or informative the provided frame is in the context of the given question.

**Instructions for Analysis and Response**

In your analysis, please perform the following steps to finish your evaluation:

1. Describe the visual content of the sampled frame, focusing on elements relevant to the question, if such elements are present.

2. Assign a relevance reward between 0 and 100 based on: (1) The sampled frame's direct usefulness in answering the question (2) Whether the frame suggests that adjacent frames might provide additional information that help answer the question more effectively.

Please provide your answer in the following format: {"description": str, "reward": int}.

**User Input**

Video Duration: <<<duration>>> seconds
Sampled Frame Timestamp: <<<timestamp>>> seconds
Question: <<<question>>>"""


class VideoDataset(Dataset):
    def __init__(self, json_file, video_dir, dataset_type):
        self.video_dir = video_dir
        self.dataset_type = dataset_type
        
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        processed_keys = {'videoid', 'question', 'r_frame_idx', 'boundaries', 'reward', 'query_type', 'video_path'}
        self.needs_processing = any(k not in processed_keys for k in self.data[0].keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if not self.needs_processing:
            return item

        question = item['question'].strip()
        video_path, videoid = '', ''

        if self.dataset_type == 'mlvu':
            name = item['video_name']
            video_path = os.path.join(self.video_dir, name)
            videoid = name.replace('.mp4', '')

        elif self.dataset_type == 'videomme':
            videoid = item['videoID']
            video_path = os.path.join(self.video_dir, f"{videoid}.mp4")
            if options := item.get('options'):
                question += '\n' + '\n'.join(options)

        elif self.dataset_type == 'longvideobench':
            name = item['video_path']
            video_path = os.path.join(self.video_dir, name)
            videoid = name.replace('.mp4', '')
            if candidates := item.get('candidates'):
                fmt_opts = [f"({chr(65+i)}) {opt}" for i, opt in enumerate(candidates)]
                question += '\n' + '\n'.join(fmt_opts)

        return {'video_path': video_path, 'videoid': videoid, 'question': question}
import logging
import random
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim


from audiocraft.models import MusicGen

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# theme word combination options
THEMES_WORD_ELEMENTS = [
    "Love", "Heartbreak", "Nature", "Exploration", "Sci-fi", "Futuristic", "Motivation", "Resilience", "Storytelling", "Legends",
    "Philosophy", "Romantic", "Thoughts", "Sadness", "Melancholy", "Deep", "Joy", "Celebration", "Mystery", "Dark", "Light", "Fantasy",
    "Ecstasy", "Nostalgia", "Anger", "Serenity", "Anxiety", "Hope", "Envy", "Awe", "Underwater", "Desert", "Jungle", "Space Station",
    "Carnival", "Graveyard", "Medieval", "Renaissance", "Ancient Egypt", "Japanese Samurai", "African Safari", "Running", "Dancing",
    "Fighting", "Reading", "Flying", "Adventure", "Loneliness", "Magic", "Victory", "Reflection", "Surprise", "Tranquility", "Urgency",
    "Melancholy", "Nostalgia", "Epic", "Battle", "Cyberpunk", "City", "Shine", "Sunset", "Dark", "Forest", "Mystery", "Space", "Odyssey",
    "Chill", "Lo-Fi", "Vibes", "Jazz", "Lounge", "Night", "Day", "Ancient", "Ruins", "Exploration", "Medieval", "Folk", "Dance",
    "Dream", "Awakening", "Journey", "Destiny", "Time", "Eternity", "Universe", "Cosmic", "Stars", "Moon", "Sun", "Rain", "Snow",
    "Wind", "Fire", "Water", "Earth", "Sky", "Ocean", "River", "Mountain", "Valley", "Flower", "Tree", "Leaf", "Bird", "Animal",
    "Whisper", "Silence", "Echo", "Shadow", "Light", "Color", "Sound", "Harmony", "Chaos", "Balance", "Peace", "War", "Life", "Death",
    "Spirit", "Soul", "Heart", "Mind", "Body", "Strength", "Weakness", "Courage", "Fear", "Passion", "Desire", "Memory", "Future",
    "Present", "Past", "Childhood", "Adulthood", "Wisdom", "Knowledge", "Truth", "Lie", "Hope", "Despair", "Faith", "Doubt",
    "Change", "Growth", "Decay", "Creation", "Destruction", "Freedom", "Imprisonment", "Justice", "Injustice", "Love Song",
    "Ballad", "Anthem", "Hymn", "Lullaby", "Elegy", "Ode", "Symphony", "Concerto", "Sonata", "Etude", "Nocturne", "Waltz", "Tango",
    "Rumba", "Samba", "Cha-cha", "Swing", "Blues", "Rock", "Pop", "Hip-hop", "Electronic", "Classical", "World Music", "Ambient",
    "Minimalist", "Experimental", "Avant-garde", "Indie", "Alternative", "Underground", "Mainstream", "Commercial", "Independent",
    "Art", "Music", "Dance", "Theater", "Film", "Literature", "Poetry", "Painting", "Sculpture", "Architecture", "Science",
    "Technology", "Engineering", "Mathematics", "History", "Geography", "Culture", "Society", "Politics", "Economics", "Religion",
    "Philosophy", "Psychology", "Sociology", "Anthropology", "Education", "Health", "Environment", "Sustainability", "Innovation",
    "Progress", "Revolution", "Evolution", "Transformation", "Harmony", "Dissonance", "Contrast", "Balance", "Symmetry", "Asymmetry",
    "Repetition", "Variation", "Improvisation", "Composition", "Performance", "Audience", "Concert", "Festival", "Club", "Studio",
    "Stage", "Backstage", "Microphone", "Instrument", "Voice", "Melody", "Harmony", "Rhythm", "Tempo", "Dynamics", "Timbre", "Texture",
    "Form", "Structure", "Style", "Genre", "Era", "Movement", "Influence", "Inspiration", "Creativity", "Expression", "Communication",
    "Emotion", "Feeling", "Mood", "Atmosphere", "Story", "Narrative", "Theme", "Motif", "Symbol", "Metaphor", "Allegory", "Irony",
    "Humor", "Wit", "Sarcasm", "Paradox", "Mystery", "Suspense", "Drama", "Comedy", "Tragedy", "Romance", "Adventure", "Fantasy",
    "Horror", "Thriller", "Science Fiction", "Historical Fiction", "Contemporary", "Classic", "Modern", "Postmodern", "Abstract",
    "Surreal", "Realism", "Naturalism", "Romanticism", "Symbolism", "Expressionism", "Dadaism", "Surrealism", "Modernism",
    "Postmodernism", "Minimalism", "Conceptual Art", "Performance Art", "Installation Art", "Video Art", "Digital Art", "Pop Art",
    "Op Art", "Land Art", "Environmental Art", "Social Commentary", "Political Satire", "Personal Reflection", "Spiritual Journey",
    "Inner Peace", "Outer Chaos", "Human Condition", "Universal Themes", "Timeless Truths", "Ephemeral Beauty", "Fleeting Moments",
    "Precious Memories", "Lost Loves", "Unspoken Words", "Hidden Meanings", "Secret Desires", "Dreams", "Nightmares", "Hopes", "Fears",
    "Joys", "Sorrows", "Triumphs", "Failures", "Beginnings", "Endings", "Life", "Death", "Creation", "Destruction", "Order", "Chaos",
    "Light", "Shadow", "Good", "Evil", "Love", "Hate", "Peace", "War", "Freedom", "Imprisonment", "Justice", "Injustice",
    "Truth", "Lies", "Faith", "Doubt", "Certainty", "Uncertainty", "Change", "Growth", "Decay", "Renewal", "Progress", "Regression",
    "Evolution", "Transformation", "Harmony", "Dissonance", "Balance", "Imbalance", "Symmetry", "Asymmetry", "Repetition", "Variation",
    "Improvisation", "Composition", "Performance", "Audience", "Concert", "Festival", "Club", "Studio", "Stage", "Backstage",
    "Microphone", "Instrument", "Voice", "Melody", "Harmony", "Rhythm", "Tempo", "Dynamics", "Timbre", "Texture", "Form", "Structure",
    "Style", "Genre", "Era", "Movement", "Influence", "Inspiration", "Creativity", "Expression", "Communication", "Emotion", "Feeling",
    "Mood", "Atmosphere", "Story", "Narrative", "Theme", "Motif", "Symbol", "Metaphor", "Allegory", "Irony", "Humor", "Wit", "Sarcasm",
    "Paradox", "Mystery", "Suspense", "Drama", "Comedy", "Tragedy", "Romance", "Adventure", "Fantasy", "Horror", "Thriller",
    "Science", "Fiction", "Historical", "Contemporary", "Classic", "Modern", "Postmodern", "Abstract", "Surreal", "Realism",
    "Naturalism", "Romanticism", "Symbolism", "Expressionism", "Dadaism", "Surrealism", "Modernism", "Postmodernism", "Minimalism",
    "Conceptual", "Art", "Performance", "Installation", "Video", "Digital", "Pop", "Op", "Land", "Environmental", "Social", "Commentary",
    "Political", "Satire", "Personal", "Reflection", "Spiritual", "Journey", "Inner", "Outer", "Human", "Condition", "Universal",
    "Themes", "Timeless", "Truths", "Ephemeral", "Beauty", "Fleeting", "Moments", "Precious", "Memories", "Lost", "Loves",
    "Unspoken", "Words", "Hidden", "Meanings", "Secret", "Desires", "Dreams", "Nightmares", "Hopes", "Fears", "Joys", "Sorrows",
    "Triumphs", "Failures", "Beginnings", "Endings", "Life", "Death", "Creation", "Destruction", "Order", "Chaos", "Light", "Shadow",
    "Good", "Evil", "Love", "Hate", "Peace", "War", "Freedom", "Imprisonment", "Justice", "Injustice", "Truth", "Lies", "Faith",
    "Doubt", "Certainty", "Uncertainty", "Change", "Growth", "Decay", "Renewal", "Progress", "Regression", "Evolution", "Transformation",
    "Harmony", "Dissonance", "Balance", "Imbalance", "Symmetry", "Asymmetry", "Repetition", "Variation"
    ]


# script to generate 5k random music title, in order to generate synthetic midi output for student model training
def random_generate_music_title(elem_array):
    # random and max length for music title
    random_text_len = random.randint(1, 3)
    text = ""
    for i in range(random_text_len):
        # random music title generation
        random_text_elem = random.choice(elem_array)
        text += ((random_text_elem + " ") if random_text_len - i > 1 else random_text_elem)
    return text


# saving the music titles into a file
def save_music_titles(music_quantity, output_path):
    with open(f"{output_path}/music_title.txt", "w", encoding="utf-8") as file:
        for _ in range(music_quantity):
            text = random_generate_music_title(THEMES_WORD_ELEMENTS)
            file.write(text)
            file.write("\n")


# generate a batch of music
def generate_music_batch(teacher_model, music_titles):
    teacher_midis = []
    for music_title in music_titles:
        midi_tokens = teacher_model.generate([music_title], progress=True)
        midi_tokens_cpu = midi_tokens.cpu()
        teacher_midis.append(midi_tokens_cpu)
    return teacher_midis


# generate the music dataset
def generate_music_dataset(output_path, batch_size=10):
    # activating the teacher model from musicgen
    teacher_model = MusicGen.get_pretrained("facebook/musicgen-small").to(device)

    # input of music titles
    music_titles = []

    # open saved music titles files
    with open(f"{output_path}/music_title.txt", "r", encoding="utf-8") as file:
        for line in file:
            music_title = line.strip()
            music_titles.append(music_title)

    teacher_midis = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(music_titles), batch_size):
            batch = music_titles[i:i + batch_size]
            futures.append(executor.submit(generate_music_batch, teacher_model, batch))

        for future in as_completed(futures):
            teacher_midis.extend(future.result())

    try:
        # save the output in a tensor shape for futher training usage
        torch.save({"title": music_titles, "midi": teacher_midis}, f"{output_path}/teacher_midis.pt")
    except Exception as e:
        print(f"Error by saving teacher midis: ", e)


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # save the music titles
    save_music_titles(args.music_quantity, args.output_dir)

    # generate the music dataset
    generate_music_dataset(args.output_dir, args.batch_size)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/musicgen-small", help="Model name or path")
    parser.add_argument("--music_quantity", type=int, default=100, help="Number of music titles to generate")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size to generate music")
    parser.add_argument("--output_dir", type=str, default="./lyrics2music-dataset", help="Directory to save the dataset")
    args = parser.parse_args()

    main(args)
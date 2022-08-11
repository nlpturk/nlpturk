import os
import json
from pathlib import Path
from typing import Iterator, Union, Tuple, List, Dict, Any


class FS:
    """Class for file I/O and utilities.
    """
    @staticmethod
    def split_path(filepath: Union[str, Path]) -> Tuple[str, str, str]:
        """Extract file extension, file name and parent path.

        Args:
            filepath (Union[str, Path]): File path.

        Returns:
            Tuple[str, str, str]: Extracted extension, filename and parent path. 
        """
        parent, basename = os.path.split(str(filepath).rstrip(os.sep))
        filename, ext = os.path.splitext(basename)
        ext = ext[1:].strip().lower()
        return ext, filename, parent

    @staticmethod
    def read_json(filepath: Union[str, Path]) -> Union[Dict[str, Any], List[Any]]:
        """Read json file.

        Args:
            filepath (Union[str, Path]): File path.

        Returns:
            Union[Dict[str, Any], List[Any]]: File content.
        """
        with open(filepath, encoding='utf-8') as f:
            content = json.load(f)
        return content

    @staticmethod
    def write_json(
        content: Union[str, Dict[Any, Any], List[Any]],
        filepath: Union[str, Path],
        append: bool = False
    ) -> None:
        """Dump json file.

        Args:
            content (Union[Dict[Any, Any], List[Any]]): Content to be dumped.
            filepath (Union[str, Path]): File path.
            append (bool): Append to the existing file or not. Update key values 
                if file is a json dictionary and key exists. Defaults to False.
        """
        if not isinstance(content, (str, dict, list)):
            raise ValueError('Content must be a string, dictionary or list.')
        content = json.loads(content) if isinstance(content, str) else content
        if append and os.path.isfile(filepath):
            with open(filepath, encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                if not isinstance(content, dict):
                    raise ValueError('Content must be a dictionary.')
                data.update(content)
            elif isinstance(data, list):
                data.append(content)
            else:
                raise ValueError('An error occured while reading json file.')
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
        else:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False)

    @staticmethod
    def read_jsonl(filepath: Union[str, Path]) -> List[Union[Dict[str, Any], List[Any]]]:
        """Read jsonlines file.

        Args:
            filepath (Union[str, Path]): File path.

        Returns:
            Union[Dict[str, Any], List[Any]]: File content.
        """
        content = []
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                content.append(json.loads(line.strip()))
        return content

    @staticmethod
    def write_jsonl(
        content: List[Union[Dict[Any, Any], List[Any]]],
        filepath: Union[str, Path],
        append: bool = False
    ) -> None:
        """Dump jsonlines file.

        Args:
            content (List[Union[Dict[Any, Any], List[Any]]]): Content to be dumped.
            filepath (Union[str, Path]): File path.
            append (bool): Append to the existing file or not. Defaults to False.
        """
        if not isinstance(content, list) or not all(isinstance(e, (dict, list)) for e in content):
            raise ValueError('Content must be a list of dictionaries or lists.')
        if append and os.path.isfile(filepath):
            with open(filepath, 'a+', encoding='utf-8') as f:
                for line in content:
                    json_record = json.dumps(line, ensure_ascii=False)
                    f.write(f'\n{json_record}')
        else:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                for line in content:
                    json_record = json.dumps(line, ensure_ascii=False)
                    f.write(f'{json_record}\n')
                f.truncate(f.tell() - len(os.linesep))

    @staticmethod
    def read_conll(filepath: Union[str, Path]) -> Iterator[List[Union[str, List[str]]]]:
        """Read conll file.

        Args:
            filepath (Union[str, Path]): File path.

        Yields:
            Iterator[List[Union[str, List[str]]]]: Yield conll file content,
                one for each sentence. 
        """
        with open(filepath, encoding="utf-8") as f:
            sents = f.read().strip().split('\n\n')
        for sent in sents:
            lines = []
            for line in [l.strip() for l in sent.strip().split('\n')]:
                lines.append(line if line.startswith('#') else line.split())
            yield lines

    @staticmethod
    def parse_conllu(
        filepath: Union[str, Path],
        min_tokens: int = 5
    ) -> List[Dict[str, List[str]]]:
        """Read conllu file, merge subtokens and normalize lemmas. 
        Lemma cases are inconsistent among treebanks.

        Args:
            filepath (Union[str, Path]): Conllu file path.
            min_tokens (int, optional): Minimum number of tokens the sentence should have.

        Returns:
            List[List[str]]: List of parsed sentences.
        """
        def lower(text: str) -> str:
            return text.replace('I', 'ı').replace('İ', 'i').lower()

        def isupper(text: str) -> str:
            return text == text.replace('i', 'İ').upper()

        def normalize_lemma(token, lemma):
            i, prefix = 1, ''
            while i <= len(lemma) and lower(token[:i]) == lower(lemma[:i]):
                prefix = token[:i]
                i += 1
            return prefix + lemma[len(prefix):]

        sents = []
        for lines in FS.read_conll(filepath):
            exclude = []
            for idx, line in enumerate(lines):
                if isinstance(line, str):
                    exclude.append(idx)
                    continue
                # merge subtokens
                if '-' in line[0]:
                    lines[idx+1][1] = line[1]
                    exclude.extend([idx, idx+2])
                # normalize lemma
                lines[idx][2] = normalize_lemma(line[1], line[2])
            lines = [l for idx, l in enumerate(lines) if idx not in exclude]
            if len(lines) >= min_tokens:
                # some sents are all uppercase, lower tokens and lemmas
                if all(isupper(l[1]) for l in lines):
                    for idx, line in enumerate(lines):
                        lines[idx][1] = lower(line[1])
                        lines[idx][2] = lower(line[2])
                sent = {k: [] for k in ('words', 'lemmas', 'poses', 'morphs')}
                for line in lines:
                    _, word, lemma, pos, _, morph, *_ = line
                    sent['words'].append(word)
                    sent['lemmas'].append(lemma)
                    sent['poses'].append(pos)
                    sent['morphs'].append(morph)
                sents.append(sent)
        return sents

    @staticmethod
    def read(filepath: Union[str, Path]) -> Union[str, Dict[str, Any], List[Any]]:
        """Read content from file.

        Args:
            filepath (Union[str, Path]): File path.

        Returns:
            Union[str, Dict[str, Any], List[Any]]: File content.
        """
        ext, *_ = FS.split_path(filepath)
        if not os.path.isfile(filepath):
            raise ValueError(f'File does not exist at path "{filepath}".')

        if ext.startswith('conll'):
            return FS.read_conll(filepath)
        elif ext == 'json':
            return FS.read_json(filepath)
        elif ext == 'jsonl':
            return FS.read_jsonl(filepath)
        else:
            with open(filepath, encoding='utf-8') as f:
                content = f.read()
            return content

    @staticmethod
    def to_disk(
        content: Union[str, Dict[Any, Any], List[Any]],
        filepath: Union[str, Path],
        append: bool = False
    ) -> None:
        """Write content to file.

        Args:
            content (Union[str, Dict[Any, Any], List[Any]]): File content to be written.
            filepath (Union[str, Path]): File path.
            append (bool, optional): Append to the existing file or not. Defaults to False.
        """
        ext, *_ = FS.split_path(filepath)
        if ext == 'json':
            FS.write_json(content, filepath, append=append)
        elif ext == 'jsonl':
            FS.write_jsonl(content, filepath, append=append)
        else:
            if append and os.path.isfile(filepath):
                with open(filepath, 'a+', encoding='utf-8') as f:
                    f.write(f'\n{content}')
            else:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)

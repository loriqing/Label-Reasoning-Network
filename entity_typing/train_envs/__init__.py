#!/usr/bin/env python
# -*- coding:utf-8 -*-

from entity_typing.train_envs.typing_env import TypingTrainEnv
from entity_typing.train_envs.bert_typing_env import BertTypingTrainEnv
from entity_typing.train_envs.seq_typing_env import SeqTypingTrainEnv
from entity_typing.train_envs.mem_typing_env import MemTypingTrainEnv
from entity_typing.train_envs.elmo_typing_env import ElmoTypingTrainEnv

train_env_map = {
    'typing': TypingTrainEnv,
    'seq_typing': SeqTypingTrainEnv,
    'mem_typing': MemTypingTrainEnv,
    'bert_typing': BertTypingTrainEnv,
    'elmo_typing': ElmoTypingTrainEnv,
}

if __name__ == "__main__":
    pass

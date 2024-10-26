function cl=cleanRequired(seqFolder)

cl =~isempty(strfind(seqFolder,'AAAA')) || ~isempty(strfind(seqFolder,'BBBB')) || ~isempty(strfind(seqFolder,'CCCC'));

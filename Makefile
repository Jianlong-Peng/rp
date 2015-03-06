dirs = svm partition thread em

.PHONY: all clean $(dirs)

all: $(dirs)

clean: $(dirs)

$(dirs):
	$(MAKE) -C $@ $(MAKECMDGOALS)


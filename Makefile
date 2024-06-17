BUILD_DIR := build
TEST_DIR  := tests
UNIT_TEST ?= test_vec_copy_f32


.PHONY: build clean test

build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. && make

test: build
	@python3 $(TEST_DIR)/$(UNIT_TEST).py

clean:
	@rm -rf $(BUILD_DIR)
BUILD_DIR  := build
TEST_DIR   := tests
BENCH_DIR  := benchs
UNIT_TEST  ?= test_2d_tile_copy
BENCHMARK  ?= bench_flash_attn_f32


.PHONY: build clean test

build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. && make

test: build
	@python3 $(TEST_DIR)/$(UNIT_TEST).py

bench_py: build
	@python3 $(BENCH_DIR)/python/$(BENCHMARK).py

clean:
	@rm -rf $(BUILD_DIR)
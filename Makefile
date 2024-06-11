BUILD_DIR := build


.PHONY: build clean

build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. && make

clean:
	@rm -rf $(BUILD_DIR)
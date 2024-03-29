import os
from datetime import datetime
from pathlib import Path

import frontmatter
from slugify import slugify

input_base_path = Path("./.process_data/blog-posts")
output_base_path = Path("./src/content/blog")

os.system("git clone https://github.com/eduardo-escoto/blog-posts.git ./.process_data/blog-posts")


if __name__ == "__main__":
    for md_path in input_base_path.glob("*.md"):
        print(f"processing: {md_path.stem}")

        if md_path.stem != "README":
            output_path = output_base_path / Path(f"{slugify(md_path.stem)}.md")

            print(f"old path:{md_path}")
            print(f"new path:{output_path}")

            post = frontmatter.load(md_path)

            if "title" not in post.metadata.keys():
                post.metadata["title"] = output_path.stem

            if "date" not in post.metadata.keys():
                post.metadata["date"] = datetime.today().strftime("%Y-%m-%d")

            if "summary" not in post.metadata.keys():
                post.metadata["summary"] = post.content[:100] + "..."

            with open(output_path, "wb") as outfile:
                frontmatter.dump(post, outfile)

os.system("rm -rf ./.process_data/blog-posts")
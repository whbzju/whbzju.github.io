import fs from "node:fs";
import path from "node:path";

const root = path.resolve(".");
const baseBlogPosts = JSON.parse(
  fs.readFileSync(path.join(root, "src/data/blog-posts.json"), "utf8")
);
const generatedZhPosts = JSON.parse(
  fs.readFileSync(path.join(root, "src/data/translated-blog-posts.zh.json"), "utf8")
);
const translatedTs = fs.readFileSync(
  path.join(root, "src/data/translated-blog-posts.ts"),
  "utf8"
);

// zhihu file is a JS module: `export const zhihuBlogPosts = [ ... ];`
let zhihuRaw = fs.readFileSync(path.join(root, "src/data/zhihu-blog-posts.ts"), "utf8");
zhihuRaw = zhihuRaw.replace(/^\s*export const zhihuBlogPosts\s*=\s*/, "").trim();
zhihuRaw = zhihuRaw.replace(/\bas const\s*;?\s*$/, "").trim();
if (zhihuRaw.endsWith(";")) zhihuRaw = zhihuRaw.slice(0, -1);
zhihuRaw = zhihuRaw.trim();
const zhihuBlogPosts = JSON.parse(zhihuRaw);

const all = [
  ...baseBlogPosts.map((p) => ({ ...p, _file: "blog-posts.json" })),
  ...zhihuBlogPosts.map((p) => ({ ...p, _file: "zhihu-blog-posts.ts" })),
  ...generatedZhPosts.map((p) => ({ ...p, _file: "translated-blog-posts.zh.json" })),
];

const textOf = (b) => {
  if (!b) return "";
  if (b.type === "ul" || b.type === "ol") return (b.items || b.itemsEn || b.itemsZh || []).join(" ");
  return b.html || b.htmlEn || b.htmlZh || b.caption || b.captionEn || b.captionZh || "";
};
const hasCJK = (s) => /[\u4e00-\u9fff]/.test(s || "");
const stripLinks = (s) => (s || "").replace(/<a[^>]*>.*?<\/a>/g, "");

console.log("slug | file | structure | #blocks | EN status");
console.log("---");
for (const p of all) {
  let structure, enIssues = [], nBlocks = 0;
  if (p.blocksEn && p.blocksZh) {
    structure = "dual";
    nBlocks = p.blocksEn.length;
    p.blocksEn.forEach((b, i) => {
      if (hasCJK(stripLinks(textOf(b)))) enIssues.push(`EN block ${i} has CJK`);
    });
  } else if (p.blocksZh && p._file === "translated-blog-posts.zh.json") {
    structure = "generated-zh";
    nBlocks = p.blocksZh.length;
    if (hasCJK(translatedTs)) {
      enIssues.push("translated-blog-posts.ts contains CJK; inspect English edition");
    }
  } else if (p.blocks) {
    structure = "per-block";
    nBlocks = p.blocks.length;
    p.blocks.forEach((b, i) => {
      if (b.type === "image") {
        if (b.captionZh && !b.captionEn) enIssues.push(`block ${i} image missing captionEn`);
        return;
      }
      if (b.type === "ul" || b.type === "ol") {
        if ((b.itemsZh?.length) && !(b.itemsEn?.length)) enIssues.push(`block ${i} list missing itemsEn`);
        return;
      }
      const hasZh = b.htmlZh !== undefined && b.htmlZh !== "";
      const hasEn = b.htmlEn !== undefined && b.htmlEn !== "";
      if (hasZh && !hasEn) enIssues.push(`block ${i} missing htmlEn`);
      else if (hasEn && hasCJK(stripLinks(b.htmlEn))) enIssues.push(`block ${i} htmlEn still has CJK`);
    });
  } else {
    structure = "UNKNOWN";
  }
  const status = enIssues.length === 0 ? "OK" : enIssues.length + " issue(s)";
  console.log(`${p.slug} | ${p._file} | ${structure} | ${nBlocks} | ${status}`);
  enIssues.slice(0, 6).forEach((x) => console.log(`    - ${x}`));
}

import baseBlogPosts from "./blog-posts.json";
import { zhihuBlogPosts } from "./zhihu-blog-posts";
import { translatedBlogPosts } from "./translated-blog-posts";

export const allBlogPosts = [...baseBlogPosts, ...zhihuBlogPosts, ...translatedBlogPosts];

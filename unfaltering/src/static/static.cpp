// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "unfaltering/static.hpp"
#include <cerrno>
#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <vector>
namespace uf {
Fd open_root(const std::filesystem::path &path) {
  Fd fd(::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  if (!fd)
    throw std::runtime_error("cannot open configured root (symlink roots are prohibited)");
  return fd;
}
static int hex(char c) {
  if (c >= '0' && c <= '9')
    return c - '0';
  if (c >= 'a' && c <= 'f')
    return c - 'a' + 10;
  if (c >= 'A' && c <= 'F')
    return c - 'A' + 10;
  return -1;
}
Resource open_resource(int root, std::string_view target) {
  Resource r;
  auto error = [&](int s) {
    r.status = s;
    return std::move(r);
  };
  auto encoded = target.substr(0, target.find('?'));
  std::string decoded;
  for (std::size_t i = 0; i < encoded.size(); ++i) {
    unsigned char c = static_cast<unsigned char>(encoded[i]);
    if (c == '%') {
      if (i + 2 >= encoded.size())
        return error(400);
      int a = hex(encoded[i + 1]), b = hex(encoded[i + 2]);
      if (a < 0 || b < 0)
        return error(400);
      c = static_cast<unsigned char>(a * 16 + b);
      i += 2;
    }
    if (c == 0 || c == '\\' || c < 32 || c == 127 || c == '%')
      return error(403);
    decoded += static_cast<char>(c);
  }
  if (decoded.empty() || decoded.front() != '/')
    return error(400);
  std::vector<std::string> segments;
  std::size_t pos = 1;
  while (pos < decoded.size()) {
    auto e = decoded.find('/', pos);
    auto segment = decoded.substr(pos, e == decoded.npos ? decoded.size() - pos : e - pos);
    if (segment == ".." || segment == ".")
      return error(403);
    if (!segment.empty()) {
      if (segment.size() > 255)
        return error(403);
      segments.push_back(segment);
    }
    if (e == decoded.npos)
      break;
    pos = e + 1;
  }
  Fd directory(::fcntl(root, F_DUPFD_CLOEXEC, 0));
  if (!directory)
    return error(500);
  for (std::size_t i = 0; i < segments.size(); ++i) {
    bool last = i + 1 == segments.size();
    Fd next(::openat(directory.get(), segments[i].c_str(),
                     O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK | (last ? 0 : O_DIRECTORY)));
    if (!next) {
      int e = errno;
      return error((e == ELOOP || e == EACCES || e == ENOTDIR) ? 403 : 404);
    }
    struct stat st {};
    if (::fstat(next.get(), &st) < 0)
      return error(500);
    if (!last && !S_ISDIR(st.st_mode))
      return error(403);
    if (last && S_ISREG(st.st_mode)) {
      r.file = std::move(next);
      break;
    }
    if (!S_ISDIR(st.st_mode))
      return error(403);
    directory = std::move(next);
  }
  if (!r.file) {
    r.file.reset(
        ::openat(directory.get(), "index.html", O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));
    if (!r.file)
      return error(errno == ELOOP || errno == EACCES ? 403 : 404);
    decoded += "/index.html";
  }
  struct stat st {};
  if (::fstat(r.file.get(), &st) < 0)
    return error(500);
  if (!S_ISREG(st.st_mode) || st.st_size < 0)
    return error(403);
  r.length = static_cast<std::size_t>(st.st_size);
  r.status = 200;
  auto ext =
      decoded.substr(decoded.rfind('.') == decoded.npos ? decoded.size() : decoded.rfind('.'));
  if (ext == ".html" || ext == ".htm")
    r.mime = "text/html; charset=utf-8";
  else if (ext == ".css")
    r.mime = "text/css; charset=utf-8";
  else if (ext == ".js")
    r.mime = "text/javascript; charset=utf-8";
  else if (ext == ".json")
    r.mime = "application/json";
  else if (ext == ".png")
    r.mime = "image/png";
  else if (ext == ".jpg" || ext == ".jpeg")
    r.mime = "image/jpeg";
  else if (ext == ".svg")
    r.mime = "image/svg+xml";
  else if (ext == ".txt")
    r.mime = "text/plain; charset=utf-8";
  else
    r.mime = "application/octet-stream";
  return r;
}
} // namespace uf
